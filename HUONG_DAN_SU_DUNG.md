# Hướng dẫn chat realtime và Query trace

## 1. Trạng thái bàn giao

Phiên bản trên nhánh `Vu` bổ sung tiến độ truy vấn, trả lời từng dòng đã kiểm tra,
worker nền và trang thống kê. Theo yêu cầu, chưa cài môi trường, chạy kiểm thử,
build, dịch vụ hoặc benchmark trong lần triển khai này. Các bước dưới đây là
hướng dẫn để chạy sau; không phải kết quả kiểm chứng. Chưa có số liệu hiệu năng mới.

## 2. Chuẩn bị môi trường

Thực hiện trong thư mục `entertainment_dss`. Cần Python 3.12, Docker Desktop,
Ollama và dữ liệu dự án theo README. Streamlit tối thiểu 1.37 để hỗ trợ fragment
tự cập nhật. Cần MySQL và đăng nhập để theo dõi cùng tài khoản từ nhiều tab.

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pip install -e . --no-deps
ollama pull qwen3-embedding:4b
ollama pull qwen3:4b
.\.venv\Scripts\python.exe -m entdec auth db-up
.\.venv\Scripts\python.exe -m entdec auth init-db
```

`auth db-up` tạo `.env` với thông tin local nếu chưa có. Nếu đã có `.env`,
bổ sung biến MySQL, `ENTDEC_DB_URL`, `ENTDEC_SECRET_KEY` theo `.env.example`;
không thay khoá mã hoá của tài khoản đã đăng ký. Giữ `ENTDEC_AUTH_DISABLED=0`.
Không commit `.env`. Ollama phải đang chạy; đọc README để chuẩn bị Neo4j và dữ liệu.

### Nâng cấp database đã có

Dừng UI trước khi nâng cấp và sao lưu MySQL theo quy trình của bạn. Chạy lại
`python -m entdec auth init-db` bằng Python trong venv. Schema 3 thêm `query_runs`
và `query_events`; không xoá lịch sử chat. Trace cũ trong metadata tin nhắn được
nhập theo `trace_id`, không nhập trùng. Các thời gian chưa từng đo vẫn thiếu,
không được suy ra hoặc điền số 0. Sau đó khởi động lại UI.

### Chuẩn bị dữ liệu và mở ứng dụng

Nếu chưa có artefact, làm theo mục dữ liệu trong README rồi chạy pipeline;
không cần chạy lại toàn bộ nếu đã có artefact phù hợp profile.

```powershell
.\.venv\Scripts\python.exe -m entdec all --profile tiny
.\.venv\Scripts\python.exe -m entdec ui --profile tiny
```

Truy cập `http://localhost:8501`. Đăng ký, xác nhận TOTP và lưu mã khôi phục;
sau đó đăng nhập. Có thể chạy `ui --profile tiny --no-graph` để dùng Parquet
khi không sử dụng Neo4j, nhưng vẫn cần dữ liệu và artefact local.

## 3. Sử dụng Chat

1. Chọn **New chat** hoặc mở hội thoại đã lưu.
2. Gửi câu hỏi, ví dụ: “Gợi ý game có cốt truyện giống Inception”.
3. Theo dõi khung tiến độ: chờ worker, embedding, bộ nhớ, ý định, tham chiếu,
   truy xuất, xếp hạng, bằng chứng, sinh trả lời và lưu kết quả.
4. Mỗi dòng trả lời xuất hiện sau khi hoàn thành và đối chiếu với ứng viên.
   Đây là streaming theo dòng, không phải hiệu ứng gõ chữ sau khi đã sinh xong.
5. Khi hoàn tất, câu trả lời và lịch sử được giữ lại; khung tiến độ có thể thu gọn.
6. Nhấn **Xem chi tiết truy vấn** dưới lượt đang chạy hoặc câu trả lời đã lưu.
   Chat nằm bên trái và chi tiết của lượt đó mở bên phải trong cùng phiên đăng nhập.
   Chi tiết tự cập nhật khi lượt còn chạy. Nhấn **Đóng chi tiết** để trở về bố cục một cột.

Không gửi lượt thứ hai trong cùng hội thoại khi lượt trước đang chờ/chạy.
Các hội thoại khác cũng dùng chung một worker tuần tự để hạn chế tranh chấp GPU.
Đổi trang hoặc đóng tab không huỷ công việc đã gửi, miễn tiến trình Streamlit còn chạy.

Hệ thống loại dòng sai định dạng, tiêu đề không thuộc ứng viên, tiêu đề trùng và
nội dung trong thẻ suy luận trước khi hiển thị. Việc đối chiếu ứng viên không phải
chứng minh mọi câu diễn giải của LLM đều chính xác về ngữ nghĩa.

## 4. Theo dõi trực tiếp trên Query trace

Chọn **Query trace** trong sidebar của phiên hiện tại để xem thống kê và xuất báo cáo.
Trang tự cập nhật mỗi giây. Để xem một lượt cụ thể từ bảng, sao chép ID vào ô
**Trace ID để theo dõi**.

Trang hiển thị danh sách lượt và chi tiết sự kiện: thời điểm, thời gian xử lý,
ý định, tham chiếu, nguồn Neo4j/Parquet, số ứng viên, điểm thành phần và bằng chứng.
Phần JSON chứa đầy đủ cấu hình và dữ liệu có cấu trúc để debug. Không hiển thị
prompt bí mật hay nội dung suy luận nội bộ của mô hình.

| Trạng thái | Ý nghĩa |
|---|---|
| queued | Đã tiếp nhận, đang đợi worker |
| running | Đang chạy pipeline hoặc lưu kết quả |
| complete | Đã sinh và lưu kết quả; có thể dùng template fallback |
| partial | Đã lưu các dòng hợp lệ nhưng câu trả lời chưa đầy đủ |
| error | Pipeline hoặc lưu dữ liệu thất bại; xem lỗi và bước đang xử lý |
| interrupted | Tiến trình cũ dừng khi công việc chưa kết thúc |

Chỉ dữ liệu của tài khoản hiện tại được hiển thị. Hội thoại đã xoá mềm không xuất
hiện trong truy vấn trace. Không có chức năng đọc/xoá JSONL toàn hệ thống từ trang này.

## 5. Lấy số liệu cho báo cáo

Lọc theo ngày UTC, profile, trạng thái, miền, nguồn trả lời và ID hội thoại.
Miền và nguồn trả lời thường chỉ đầy đủ khi lượt đã kết thúc, nên bộ lọc này có
thể loại lượt đang chạy. Các nút xuất sử dụng cùng snapshot dữ liệu của lần cập nhật:

- **CSV:** mỗi lượt một dòng, gồm trạng thái và các thời gian đo được.
- **JSON:** bộ lọc, thống kê, cấu hình, kết quả và toàn bộ sự kiện từng lượt.
- **Markdown:** tổng hợp chỉ số, số mẫu, profile/model và danh sách trace.

| Chỉ số (ms) | Cách hiểu |
|---|---|
| first_event_ms | Từ lúc gửi vào service đến sự kiện tiếp nhận được ghi |
| first_line_ms | Từ lúc tiếp nhận đến dòng trả lời hợp lệ đầu tiên được phát |
| queue_ms | Thời gian chờ trước khi worker bắt đầu |
| pipeline_ms | Worker bắt đầu đến khi bắt đầu lưu tin nhắn |
| persist_ms | Thời gian chuẩn bị và commit tin nhắn/bộ nhớ |
| total_ms | Từ lúc tiếp nhận đến kết quả cuối hoặc lỗi |
| embed_ms, intent_ms, retrieval_ms… | Thời gian từng giai đoạn có đo |

Các thời gian trên đo phía hệ thống, không phải thời điểm trình duyệt vẽ chữ.
Giao diện polling khoảng một giây nên có thêm độ trễ hiển thị. Không cộng total,
pipeline và các bước thành một tổng mới. Mỗi chỉ số có `n`, trung bình, p50, p95;
percentile dùng nội suy tuyến tính. Ô trống/metric không xuất hiện nghĩa là chưa có dữ liệu.
Tỷ lệ hoàn tất, lỗi, một phần và fallback dùng các lượt đã kết thúc làm mẫu số;
khi chưa có lượt kết thúc thì tỷ lệ là null.

Để báo cáo có thể tái lập, lưu file JSON gốc cùng Markdown, ghi phần cứng,
profile, model, cấu hình, ngày chạy và điều kiện cold/warm. Tách lượt warm-up,
không dùng số liệu lịch sử khoảng 53 giây/lượt như phép đo của phiên bản mới.
Streaming giúp thấy nội dung sớm hơn; không tự chứng minh tổng thời gian giảm.

## 6. API đọc trace

Khởi động API theo README và dùng bearer token của tài khoản:

```text
GET /traces?offset=0&limit=50
GET /traces/stats
GET /traces/{trace_id}
GET /traces/{trace_id}/events?after_seq=5
```

Danh sách và thống kê nhận `status`, `profile`, `conversation_id`, `domain`,
`answer_source`, `since`, `until` (ngày ISO). API kiểm tra chủ sở hữu; ID của
tài khoản khác trả not found. CLI và API chat cũ vẫn đồng bộ. Kho trace live
trong đợt này được tạo bởi worker giao diện Streamlit; chưa thêm HTTP streaming.

## 7. Xử lý sự cố và giới hạn

- **Không có bảng query_runs/query_events:** chạy `auth init-db` rồi mở lại ứng dụng.
- **MySQL không kết nối được:** kiểm tra Docker, `ENTDEC_DB_URL` và dịch vụ MySQL.
  Nếu MySQL mất kết nối đúng lúc ghi lỗi cuối cùng, trạng thái có thể còn running;
  phục hồi kết nối và khởi động lại UI để đánh dấu interrupted.
- **Ollama chậm:** xem embed/intent/answer để phân biệt thời gian chờ. GPU ít VRAM
  vẫn có thể mất thời gian nạp/đổi model; streaming không loại bỏ chi phí này.
- **Ollama lỗi trước dòng đầu:** dùng template bằng chứng nếu pipeline còn truy xuất được.
- **Stream lỗi sau vài dòng:** giữ dòng hợp lệ, đánh dấu partial; gửi lại câu hỏi
  tạo trace mới, không tự nối hay phát lại nội dung cũ.
- **Neo4j không sẵn sàng:** có thể dùng `--no-graph` với artefact local; xem nguồn
  truy xuất trong trace để biết lượt có dùng Parquet.
- **Khởi động lại server:** lượt đang chờ/chạy của tiến trình trước thành interrupted,
  không tự gửi lại. Lỗi xảy ra sau commit chat nhưng trước cập nhật trace có thể
  khiến lịch sử đã có câu trả lời trong khi trace bị gián đoạn.
- Chỉ chạy **một tiến trình Streamlit trên một database** cho worker phiên bản này.
  API có thể chạy riêng để đọc dữ liệu; chưa hỗ trợ hàng đợi phân tán nhiều worker.
- Chế độ auth disabled dùng SQLite trong session, không phù hợp demo theo dõi nhiều tab.
- Trang thống kê đọc toàn bộ trace của tài khoản để tổng hợp; dành cho quy mô project,
  chưa có phân vùng dữ liệu hay chính sách tự xoá lịch sử thống kê.

## 8. Kịch bản demo báo cáo

1. Chuẩn bị profile tiny và đăng nhập vào Chat.
2. Hỏi câu gợi ý phim/game; quay lại tiến độ trước khi có câu trả lời.
3. Nhấn **Xem chi tiết truy vấn**, rồi quan sát câu trả lời bên trái và tiến trình bên phải.
4. Hỏi tiếp “Có lựa chọn nhẹ nhàng hơn không?” để minh hoạ bộ nhớ hội thoại.
5. Mở lại chi tiết từ lịch sử, rồi vào **Query trace** ở sidebar để lọc và xuất báo cáo.
6. Khi có môi trường, thực hiện bộ câu hỏi mẫu và 20 hội thoại trong cấu hình
   đánh giá của dự án; ghi rõ số lượt thực chạy. Đợt bàn giao này chưa thực hiện bước đó.
