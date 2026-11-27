# GMRe — hướng dẫn chạy ứng dụng web

GMRe là giao diện tiếng Việt xây dựng bằng React, TypeScript và Vite. Hiện ứng dụng hỗ trợ khám phá **phim** và **game**. Âm nhạc, sách, podcast và hoạt động sẽ hiện trạng thái **Sắp ra mắt**.

## 1. Chạy giao diện để xem thử, không cần đăng nhập

### Bước 1: Cài công cụ

Cài Node.js bản LTS (bao gồm npm). Kiểm tra trong PowerShell:

```powershell
node --version
npm --version
```

### Bước 2: Mở thư mục web

Từ thư mục gốc dự án:

```powershell
cd entertainment_dss/web
```

### Bước 3: Cài thư viện và chạy ứng dụng

```powershell
npm ci
npm run dev
```

Mở địa chỉ Vite in ra trong terminal, thường là <http://127.0.0.1:5173/>. Nếu chưa tạo `web/.env`, chế độ demo được bật mặc định khi chạy `npm run dev`, không cần API, cơ sở dữ liệu hay tài khoản thật. Chọn **Dùng thử không cần tài khoản** ở trang đăng nhập để xem ứng dụng; hội thoại demo chỉ dùng dữ liệu mẫu trong trình duyệt, không lưu vào database của chatbot.

Để bật demo khi đã cấu hình `web/.env`, đặt `VITE_DEMO_MODE=true`, sau đó khởi động lại Vite.

## 2. Build và xem bản production

Trong `entertainment_dss/web`, chạy:

```powershell
npm run build
npm run preview
```

Lệnh build kiểm tra TypeScript và tạo bản tĩnh trong `web/dist`. Muốn đăng nhập bằng demo trong bản preview, đặt `VITE_DEMO_MODE=true` trước khi build vì các biến `VITE_*` được đóng gói vào ứng dụng tại thời điểm build. Không bật demo khi phát hành bản dùng tài khoản thật.

## 3. Kết nối web với backend FastAPI

Bạn đang dùng MySQL cục bộ trên cổng **3306**. File `entertainment_dss/.env` đã được tạo riêng trên máy với URL MySQL, khóa Fernet và xác thực bật; file này nằm trong `.gitignore`, không được commit.

### Bước 1: Tạo database lần đầu

Trong MySQL monitor chạy:

```sql
CREATE DATABASE IF NOT EXISTS entdec CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci;
EXIT;
```

`.env` hiện cấu hình username MySQL là `root`. Nếu bạn đăng nhập MySQL bằng username khác, sửa phần username trong `ENTDEC_DB_URL` ở file `.env` trước khi tiếp tục. Không đưa mật khẩu vào README hoặc mã nguồn.

### Bước 2: Cài backend Python (chỉ làm lần đầu)

Mở PowerShell ở thư mục workspace `cuoiki` rồi chuyển vào thư mục dự án backend. Chạy các lệnh backend từ đây để chúng tìm đúng `requirements.txt` và file `.env`:

```powershell
Set-Location .\entertainment_dss
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pip install -e . --no-deps
```

Nếu đã tạo nhầm `.venv` ở thư mục cha `cuoiki`, dòng trên sẽ tạo môi trường đúng tại `entertainment_dss\.venv`; hãy dùng Python ở đường dẫn đó cho các bước tiếp theo. Nếu `.venv` trong `entertainment_dss` đã có, bỏ qua dòng tạo môi trường. Cài dependencies không cần khởi động Ollama hay Neo4j.

### Bước 3: Tạo bảng và chạy API

Vẫn ở thư mục gốc dự án:

```powershell
.\.venv\Scripts\python.exe -m entdec auth init-db
.\.venv\Scripts\python.exe -m entdec api --profile tiny --no-graph --host 127.0.0.1 --port 8000
```

Giữ terminal này mở. Kiểm tra API tại <http://127.0.0.1:8000/docs>. `--no-graph` cùng chế độ khởi tạo AI trễ giúp các API tài khoản/hội thoại chạy mà chưa cần môi trường recommender.

### Bước 4: Chạy web với API thật

Mở terminal PowerShell thứ hai:

```powershell
Set-Location D:\Entertainment_decision\web
if (-not (Test-Path .env)) { Copy-Item .env.example .env }
npm ci
npm run dev
```

File `web/.env.example` cấu hình web dùng API thật:

```dotenv
VITE_DEMO_MODE=false
VITE_API_BASE_URL=/api
ENTDEC_API_URL=http://127.0.0.1:8000
```

Lệnh `Copy-Item` chỉ cần chạy lần đầu; nếu `web/.env` đã tồn tại, không ghi đè file đó. Mở <http://127.0.0.1:5173/> rồi đăng ký tài khoản. Luồng đăng ký yêu cầu xác nhận TOTP. Vite proxy `/api/...` đến FastAPI và bỏ tiền tố `/api`. Nếu vừa đổi file `.env`, hãy dừng rồi chạy lại Vite để nạp cấu hình.

### Nếu muốn dùng MySQL Docker của dự án

Docker Compose ánh xạ MySQL ra cổng `3307`; khi đó cập nhật URL trong `.env` từ `3306` sang `3307`, rồi chạy `docker compose up -d entdec-mysql` trước lệnh `auth init-db`. Không chạy service MySQL Docker nếu cổng máy đã được MySQL cục bộ sử dụng.

## 4. Phần web và backend cần khớp nhau

- Web không kết nối database trực tiếp: web gọi FastAPI qua `/api`. Với cấu hình xác thực bật, FastAPI lưu tài khoản, lịch sử hội thoại/tin nhắn và trạng thái tài khoản vào MySQL (`users`, `conversations`, `messages`, `account_state`). Chatbot recommender dùng Neo4j riêng cho knowledge graph/nội dung; đó là database khác theo thiết kế, không phải nơi lưu lịch sử chat.
- Web gọi các route xác thực `/auth/*`, trạng thái tài khoản `/account/state`, hội thoại `/conversations/*` và trace `/traces/*`. Trạng thái tài khoản gồm mục đã lưu, đã trải nghiệm, đánh giá, sở thích và cài đặt; backend lưu theo user trong bảng MySQL `account_state`.
- Chat web gửi lượt hỏi qua `POST /conversations/{conversation_id}/runs`; backend trả `trace_id` ngay, chạy recommender nền và lưu tiến độ/sự kiện vào cùng kho trace. Web đọc `GET /traces/{trace_id}` mỗi giây khi lượt còn chạy, sau đó tải lại hội thoại đã lưu. Nút ngay dưới câu hỏi mở khung trace cạnh Chat; có thể kéo vạch giữa để đổi tỉ lệ, kéo cạnh dưới mỗi khung để đổi chiều cao, đóng/mở lại trace và cuộn riêng từng khung. Kích thước được lưu trong tab hiện tại; trên màn hình hẹp hai khung xếp dọc. API cũ `POST /conversations/{conversation_id}/messages` vẫn được giữ cho client tương thích.
- Chế độ demo tạo trace mẫu trong trình duyệt để bạn xem khung chi tiết và tương tác mà không cần đăng nhập thật; trace demo không được lưu vào backend.
- API khởi tạo recommender khi nhận yêu cầu AI qua `/chat`, `/recommend` hoặc tin nhắn hội thoại. Nếu mô hình/dữ liệu AI chưa sẵn sàng, API trả lỗi 503 giải thích tình trạng; các route tài khoản vẫn hoạt động.
- Ảnh danh mục mẫu trên trang chủ được khai báo trong `web/src/data.ts`: phim dùng poster TMDB, game dùng artwork Steam; Stardew Valley dùng ảnh artwork đi kèm trong `web/public/image`. Ảnh gợi ý động được tải qua API `/media/{domain}/{id}`; poster phim cần `TMDB_READ_ACCESS_TOKEN` ở backend và file `links.csv`, còn game dùng artwork Steam. Nếu không tải được ảnh, web chỉ dùng ảnh dự phòng khi khớp đúng nội dung mẫu, tránh hiển thị nhầm poster.
- Backend hiện chỉ nhận domain `movie` và `game`. Các danh mục còn lại trên web được đánh dấu **Sắp ra mắt** và không gửi yêu cầu gợi ý đến backend.
- Các artwork minh họa/dự phòng nằm trong `web/public/image`; đường dẫn và nguồn ảnh danh mục được khai báo ở `web/src/data.ts`.
- Khi đăng nhập lần đầu sau cập nhật, web nhập mục đã lưu, đánh giá, sở thích và cài đặt cũ từ `localStorage` nếu MySQL chưa có trạng thái cho tài khoản; các lần sau MySQL là nguồn dữ liệu chính.
- Khi triển khai web ở môi trường khác, cấu hình máy chủ tĩnh để trả về `index.html` cho các đường dẫn SPA như `/explore`, `/saved` và `/account`. Đồng thời đặt `VITE_API_BASE_URL` phù hợp với địa chỉ API hoặc reverse proxy trước khi chạy `npm run build`.

## 5. Cấu trúc liên quan

```text
web/
├── public/image/   # ảnh minh họa cục bộ cho phim và game
├── src/App.tsx     # các trang và điều hướng
├── src/api.ts      # gọi API và chế độ demo
├── src/data.ts     # danh mục, gợi ý mẫu và đường dẫn ảnh
└── src/styles.css  # giao diện tiếng Việt
```
