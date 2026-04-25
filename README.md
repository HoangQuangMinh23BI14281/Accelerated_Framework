Giai đoạn 1: Xây dựng Nền tảng (Foundation & Memory Router)
Mục tiêu: Tạo ra một lõi engine có khả năng kiểm soát KV Cache ở cấp độ từng token và từng layer thay vì cấp độ toàn bộ sequence.

Bước 1.1: Tự code một PagedAttention tinh gọn: Đừng dùng mảng tensor tĩnh. Hãy tạo một Memory Pool cấp phát bộ nhớ VRAM theo từng "block" (ví dụ: 16 tokens/block). Điều này là nền tảng sống còn để luân chuyển token giữa 3 trạng thái của ARKV và gom batch liên tục (Continuous Batching).

Bước 1.2: Xây dựng Continuous Batching Engine: Viết một scheduler vòng lặp (while-loop). Tại mỗi bước sinh token (iteration), scheduler sẽ kiểm tra: nhánh nào đã sinh xong (gặp mã EOS hoặc bị ngắt) thì đá ra khỏi batch ngay lập tức, giải phóng block VRAM tương ứng.

Bước 1.3: Profiling Layer (Pha Prefill): Viết script tính toán Entropy và Phương sai của Attention Matrix ngay trong pha xử lý Prompt ban đầu để xuất ra một mảng VRAM_Budget[layer_id].

Giai đoạn 2: Tầng 1 - Radar Điều hướng & Quản lý Luồng (Routing)
Mục tiêu: Tích hợp bộ đo lường và cơ chế đẻ nhánh (spawn).

Bước 2.1: Triển khai LogitScope: Sửa đổi hàm forward() của mô hình. Tại bước xuất Logits, áp dụng Softmax và tính 2 chỉ số: Shannon Entropy và Varentropy.

Bước 2.2: Lập trình Rule Engine (DTS): Thiết lập logic kích hoạt rẽ nhánh dựa trên ngưỡng Entropy (thấp) và Varentropy (cao).

Bước 2.3: Xây dựng Multi-Stream Context Manager: Khi DTS kích hoạt, tạo các Side-Agents. Ở mức code, điều này có nghĩa là "nhân bản" các block index trong Paged Attention (chỉ copy con trỏ/index, không copy dữ liệu KV Cache) và gán chúng vào các CUDA Streams bất đồng bộ.

Giai đoạn 3: Tầng 2 - Lõi Tối ưu ARKV & Quantization (Thử thách nhất)
Mục tiêu: Giải quyết bài toán băng thông VRAM và Compute ngay bên trong GPU. Đây là nơi bạn sẽ phải viết Triton/CUDA custom kernel.

Bước 3.1: ARKV Heavy-Hitter Scorer: Viết kernel đánh giá tầm quan trọng của token dựa trên tổng điểm Attention nó nhận được từ các token khác.

Bước 3.2: Cơ chế Eviction (Cắt rác): Cập nhật kernel PagedAttention để nếu một token rơi vào nhóm rác, index của nó lập tức bị gỡ khỏi bảng băm (hash table), nhường chỗ trống cho token mới.

Bước 3.3: W4A16 Quantization cho K-Cache: Nén K-Cache xuống 4-bit (chấp nhận giải nén on-the-fly khi nhân QxK để cứu băng thông).

Bước 3.4: TurboQuant & Sparse V cho V-Cache: Ép nén V-Cache xuống 3-bit hoặc 4-bit. Tích hợp điều kiện: Nếu kết quả của Softmax(QxK) nhỏ hơn 10e-6, bỏ qua vòng lặp đọc và giải nén V-Cache tại token đó.

Giai đoạn 4: Tầng 3 - Kiểm định Nội dung & Hợp nhất (Validation & Injection)
Mục tiêu: Lọc ảo giác tốc độ cao và giữ lại luồng suy nghĩ tốt nhất.

Bước 4.1: Self-Consistency Checker (N-gram): Viết module so sánh độ lệch văn bản giữa các nhánh sinh ra. Nếu một nhánh có n-gram đi ngược hoàn toàn với số đông, đánh dấu cờ "Ảo giác" (Hallucination Flag).

Bước 4.2: IEW Scoring: Tính điểm Inverse-Entropy (trung bình Entropy của chuỗi). Nhánh nào bị dính cờ ảo giác ở bước 4.1 sẽ bị nhân hệ số phạt (penalty) cực nặng vào điểm này.

Bước 4.3: Referential Injection: Chọn nhánh có điểm IEW tốt nhất. Hệ thống chỉ cần xóa con trỏ KV Cache của các nhánh thua cuộc và đổi tên (rename) luồng của nhánh thắng thành Main Agent để vòng lặp đi tiếp.

Giai đoạn 5: Đóng gói và Stress Test
Mục tiêu: Đảm bảo hệ thống không bị crash khi chạy đường dài.

Bước 5.1: Lắp ráp các Giai đoạn 1-4 thành một file inference_engine.py hoàn chỉnh.

Bước 5.2: Stress Test với bài toán GSM8K hoặc MATH dataset. Bật profiling (dùng PyTorch Profiler hoặc Nsight Systems) để theo dõi vệt sử dụng VRAM (VRAM footprint) và đảm bảo không có hiện tượng OOM do phân mảnh.