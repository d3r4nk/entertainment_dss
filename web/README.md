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

Mở địa chỉ Vite in ra trong terminal, thường là <http://127.0.0.1:5173/>. Chế độ demo được bật mặc định khi chạy `npm run dev`, không cần API, cơ sở dữ liệu hay tài khoản thật. Chọn **Dùng thử không cần tài khoản** ở trang đăng nhập để xem các trang chatbot, lịch sử, nội dung đã lưu và hồ sơ. Demo dùng dữ liệu mẫu trong trình duyệt.

Nếu đã tạo `web/.env` và đặt `VITE_DEMO_MODE=false`, hãy bỏ dòng đó hoặc đổi thành `true`, sau đó khởi động lại Vite.

## 2. Build và xem bản production

Trong `entertainment_dss/web`, chạy:

```powershell
npm run build
npm run preview
```

Lệnh build kiểm tra TypeScript và tạo bản tĩnh trong `web/dist`. Muốn đăng nhập bằng demo trong bản preview, đặt `VITE_DEMO_MODE=true` trước khi build vì các biến `VITE_*` được đóng gói vào ứng dụng tại thời điểm build. Không bật demo khi phát hành bản dùng tài khoản thật.

## 3. Kết nối web với backend FastAPI

### Bước 1: Chuẩn bị backend

Backend Python nằm ở thư mục gốc `entertainment_dss`. Làm theo mục cài đặt backend trong [README của dự án](../README.md): tạo môi trường Python, cài dependencies, cấu hình `.env`, khởi động các dịch vụ cần thiết và chuẩn bị dữ liệu/profile cho recommender.

Để dùng tài khoản và lịch sử thật, cần cấu hình MySQL và `ENTDEC_DB_URL`, tạo `ENTDEC_SECRET_KEY`, đồng thời để `ENTDEC_AUTH_DISABLED=0`. API recommender có thể cần Neo4j, Ollama, dữ liệu đã xử lý và model theo profile đã chọn; xem `envcheck` và hướng dẫn backend để biết thành phần nào cần chạy trên máy của bạn.

### Bước 2: Khởi động API

Mở terminal thứ nhất ở thư mục gốc dự án. Với profile tiny đã chuẩn bị:

```powershell
.\.venv\Scripts\python.exe -m entdec api --profile tiny --host 127.0.0.1 --port 8000
```

Kiểm tra API tại <http://127.0.0.1:8000/health>. Tài liệu endpoint FastAPI thường ở <http://127.0.0.1:8000/docs>.

### Bước 3: Trỏ web đến API thật

Mở terminal thứ hai, chuyển đến `entertainment_dss/web`, tạo file môi trường nếu chưa có rồi chỉnh các giá trị:

```powershell
Copy-Item .env.example .env
```

Trong `web/.env`:

```dotenv
VITE_DEMO_MODE=false
VITE_API_BASE_URL=/api
ENTDEC_API_URL=http://127.0.0.1:8000
```

Khởi động lại Vite sau khi đổi biến môi trường:

```powershell
npm run dev
```

Vite chuyển tiếp yêu cầu `/api/...` đến FastAPI và bỏ tiền tố `/api`; ví dụ `/api/auth/me` được gửi thành `/auth/me`. Cách này tránh phải cấu hình CORS riêng khi phát triển trên địa chỉ mặc định `127.0.0.1:5173`.

### Bước 4: Tạo tài khoản thật

Trong giao diện, đăng ký bằng email và mật khẩu. Luồng hiện tại xác nhận đăng ký bằng mã TOTP, sau đó đăng nhập bằng mật khẩu và mã xác thực. Cần backend bật xác thực, MySQL hoạt động và `ENTDEC_SECRET_KEY` hợp lệ. Không dùng nút demo khi muốn kiểm tra dữ liệu tài khoản thật.

## 4. Phần web và backend cần khớp nhau

- Web gọi các route xác thực `/auth/*`, lịch sử `/conversations/*`, trò chuyện `/chat` và gợi ý `/recommend`. Payload và kiểu dữ liệu phía web được khai báo trong `src/api.ts` và `src/types.ts`; API hiện có nằm trong `src/entdec/api/main.py`.
- Backend hiện chỉ nhận domain `movie` và `game`. Các danh mục còn lại trên web được đánh dấu **Sắp ra mắt** và không gửi yêu cầu gợi ý đến backend.
- Ảnh nội dung trang chủ và thẻ gợi ý nằm trong `web/public/image`; đường dẫn hiển thị được khai báo ở `web/src/data.ts`.
- Mục đã lưu, đánh giá gu, sở thích giao diện và một số cài đặt hiện được lưu cục bộ trong trình duyệt. Muốn đồng bộ giữa thiết bị hoặc tài khoản, cần bổ sung API lưu/đọc các dữ liệu đó và thay phần `localStorage` tương ứng trong web.
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
