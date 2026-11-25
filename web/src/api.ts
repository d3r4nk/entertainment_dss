import type { ApiRecommendation, Conversation, User } from './types'
import { catalogue } from './data'

const API_BASE = import.meta.env.VITE_API_BASE_URL || '/api'
export const DEMO_MODE = import.meta.env.VITE_DEMO_MODE === 'true'
  || (import.meta.env.DEV && import.meta.env.VITE_DEMO_MODE !== 'false')
export const DEMO_TOKEN = 'gmre-demo-session'
export const isDemoToken = (token: string) => DEMO_MODE && token === DEMO_TOKEN

const demoUser: User = { id: 'gmre-demo-user', email: 'khach@gmre.demo' }
const demoItems: ApiRecommendation[] = catalogue.filter((item) => item.kind === 'movie' || item.kind === 'game').slice(0, 4).map((item) => ({
  id: item.id, domain: item.kind, title: item.title, score: item.match / 100,
  shared_themes: item.tags.slice(0, 2), shared_genres: [item.mood], evidence: item.why,
}))
const demoNow = new Date().toISOString()
let demoConversations: Conversation[] = [
  {
    id: 'gmre-demo-movie', title: 'Một bộ phim cho tối chậm rãi', created_at: demoNow, updated_at: demoNow,
    messages: [
      { id: 'gmre-demo-movie-u', role: 'user', content: 'Tối nay mình muốn xem phim nhẹ nhàng, có chút để suy ngẫm.', created_at: demoNow },
      { id: 'gmre-demo-movie-a', role: 'assistant', content: 'Mình chọn vài câu chuyện dịu dàng, hợp với một buổi tối chậm rãi nhé.', meta: { items: [demoItems[0]] }, created_at: demoNow },
    ],
  },
  {
    id: 'gmre-demo-game', title: 'Game vui để chơi cùng bạn', created_at: demoNow, updated_at: demoNow,
    messages: [
      { id: 'gmre-demo-game-u', role: 'user', content: 'Có game nào chơi thư giãn cùng bạn bè không?', created_at: demoNow },
      { id: 'gmre-demo-game-a', role: 'assistant', content: 'Đây là vài game co-op dễ bắt đầu và nhiều tiếng cười.', meta: { items: [demoItems[1], demoItems[3]] }, created_at: demoNow },
    ],
  },
]

function demoConversation(id: string) {
  const conversation = demoConversations.find((item) => item.id === id)
  if (!conversation) throw new ApiError('Không tìm thấy cuộc trò chuyện.', 404)
  return conversation
}

const messages: Record<string, string> = {
  'Authentication required.': 'Phiên đăng nhập đã hết hạn. Vui lòng đăng nhập lại.',
  'Invalid email or password.': 'Email hoặc mật khẩu chưa chính xác.',
  'Invalid authentication code.': 'Mã xác thực chưa chính xác hoặc đã hết hạn.',
  'Invalid or expired pre-authentication token.': 'Phiên xác thực đã hết hạn. Hãy đăng nhập lại.',
  'Current password is incorrect.': 'Mật khẩu hiện tại chưa chính xác.',
  'Passwords do not match.': 'Hai mật khẩu chưa trùng khớp.',
  'Password must be at least 10 characters.': 'Mật khẩu cần có ít nhất 10 ký tự.',
  'Password cannot match the email address.': 'Mật khẩu không được trùng với địa chỉ email.',
  'Choose a less common password.': 'Hãy chọn một mật khẩu khó đoán hơn.',
  'Enter a valid email address.': 'Vui lòng nhập địa chỉ email hợp lệ.',
  'Unable to register this email address.': 'Email này chưa thể đăng ký. Hãy thử đăng nhập hoặc dùng email khác.',
  'Account temporarily locked.': 'Tài khoản tạm khóa do có nhiều lần đăng nhập không thành công.',
  'Account temporarily locked': 'Tài khoản tạm khóa do có nhiều lần đăng nhập không thành công.',
  'Too many requests; try again shortly.': 'Bạn thao tác hơi nhanh. Vui lòng thử lại sau ít phút.',
}

export class ApiError extends Error {
  status: number
  constructor(message: string, status: number) {
    super(messages[message] || message || 'Đã có lỗi xảy ra. Vui lòng thử lại.')
    this.status = status
  }
}

async function request<T>(path: string, token?: string, init: RequestInit = {}): Promise<T> {
  const headers = new Headers(init.headers)
  if (init.body && !headers.has('Content-Type')) headers.set('Content-Type', 'application/json')
  if (token) headers.set('Authorization', `Bearer ${token}`)
  let response: Response
  try {
    response = await fetch(`${API_BASE}${path}`, { ...init, headers })
  } catch {
    throw new ApiError('Chưa thể kết nối máy chủ. Hãy kiểm tra API và thử lại.', 0)
  }
  if (!response.ok) {
    if (response.status === 401 && token && (path === '/auth/me' || path.startsWith('/conversations') || path.startsWith('/traces'))) {
      sessionStorage.removeItem('entdec.web.session')
      window.dispatchEvent(new Event('entdec:session-expired'))
    }
    let detail = ''
    try {
      const body = await response.json()
      detail = typeof body.detail === 'string' ? body.detail : ''
    } catch { /* Keep the friendly fallback below. */ }
    throw new ApiError(detail || 'Máy chủ chưa thể xử lý yêu cầu này.', response.status)
  }
  if (response.status === 204) return undefined as T
  return response.json() as Promise<T>
}

export const api = {
  request,
  getAccountState: (token: string) => request<{ payload: Record<string, unknown>; updated_at: string | null }>('/account/state', token),
  putAccountState: (token: string, payload: Record<string, unknown>) => request<{ payload: Record<string, unknown>; updated_at: string }>('/account/state', token, { method: 'PUT', body: JSON.stringify({ payload }) }),
  listTraces: (token: string, filters: Record<string, string> = {}) => {
    if (isDemoToken(token)) return Promise.resolve({ items: [], total: 0, offset: 0, limit: 100 })
    const query = new URLSearchParams({ offset: '0', limit: '100', ...filters })
    return request<{ items: Record<string, unknown>[]; total: number; offset: number; limit: number }>(`/traces?${query}`, token)
  },
  traceStats: (token: string, filters: Record<string, string> = {}) => isDemoToken(token) ? Promise.resolve({ count: 0, statuses: {}, metrics: {} }) : request<Record<string, unknown>>(`/traces/stats?${new URLSearchParams(filters)}`, token),
  getTrace: (token: string, id: string) => isDemoToken(token) ? Promise.reject(new ApiError('Bản demo chưa có trace.', 404)) : request<Record<string, unknown>>(`/traces/${encodeURIComponent(id)}`, token),
  getTraceEvents: (token: string, id: string, afterSeq = 0) => isDemoToken(token) ? Promise.resolve([]) : request<Record<string, unknown>[]>(`/traces/${encodeURIComponent(id)}/events?after_seq=${afterSeq}`, token),
  me: (token: string) => isDemoToken(token) ? Promise.resolve(demoUser) : request<User>('/auth/me', token),
  register: (email: string, password: string, confirm_password: string) =>
    request<{ enrollment_token: string; totp_uri: string; totp_secret: string; qr_code_png_base64: string }>(
      '/auth/register', undefined, { method: 'POST', body: JSON.stringify({ email, password, confirm_password }) },
    ),
  confirmRegistration: (enrollment_token: string, code: string) =>
    request<{ recovery_codes: string[] }>('/auth/register/confirm-2fa', undefined, {
      method: 'POST', body: JSON.stringify({ enrollment_token, code }),
    }),
  login: (email: string, password: string) =>
    request<{ preauth_token: string }>('/auth/login', undefined, {
      method: 'POST', body: JSON.stringify({ email, password }),
    }),
  login2fa: (preauth_token: string, code: string) =>
    request<{ access_token: string }>('/auth/login/2fa', undefined, {
      method: 'POST', body: JSON.stringify({ preauth_token, code }),
    }),
  logout: (token: string) => isDemoToken(token) ? Promise.resolve() : request<void>('/auth/logout', token, { method: 'POST' }),
  changePassword: (token: string, body: { current_password: string; new_password: string; confirm_password: string }) =>
    isDemoToken(token) ? Promise.reject(new ApiError('Đổi mật khẩu chỉ có trong tài khoản thật.', 400)) : request<void>('/auth/password', token, { method: 'POST', body: JSON.stringify(body) }),
  listConversations: (token: string, offset = 0, search = '') => {
    if (isDemoToken(token)) {
      const matches = demoConversations.filter((item) => item.title.toLocaleLowerCase('vi').includes(search.toLocaleLowerCase('vi')))
      return Promise.resolve({ items: matches.slice(offset, offset + 100), total: matches.length, offset, limit: 100 })
    }
    return request<{ items: Conversation[]; total: number; offset: number; limit: number }>(`/conversations?offset=${offset}&limit=100&search=${encodeURIComponent(search)}`, token)
  },
  deleteAllConversations: (token: string) => {
    if (isDemoToken(token)) { const deleted = demoConversations.length; demoConversations = []; return Promise.resolve({ deleted }) }
    return request<{ deleted: number }>('/conversations', token, { method: 'DELETE' })
  },
  getConversation: (token: string, id: string) => isDemoToken(token) ? Promise.resolve(demoConversation(id)) : request<Conversation>(`/conversations/${id}`, token),
  createConversation: (token: string, title = 'Cuộc trò chuyện mới') => {
    if (isDemoToken(token)) {
      const now = new Date().toISOString()
      const created: Conversation = { id: `gmre-demo-${Date.now()}`, title, created_at: now, updated_at: now, messages: [] }
      demoConversations = [created, ...demoConversations]
      return Promise.resolve(created)
    }
    return request<Conversation>('/conversations', token, { method: 'POST', body: JSON.stringify({ title }) })
  },
  deleteConversation: (token: string, id: string) => {
    if (isDemoToken(token)) { demoConversations = demoConversations.filter((item) => item.id !== id); return Promise.resolve() }
    return request<void>(`/conversations/${id}`, token, { method: 'DELETE' })
  },
  renameConversation: (token: string, id: string, title: string) => {
    if (isDemoToken(token)) { const item = demoConversation(id); item.title = title; return Promise.resolve(item) }
    return request<Conversation>(`/conversations/${id}`, token, { method: 'PATCH', body: JSON.stringify({ title }) })
  },
  sendMessage: (token: string, id: string, message: string) => {
    if (isDemoToken(token)) {
      const conversation = demoConversation(id)
      const now = new Date().toISOString()
      const chosen = /game|co-?op|steam/i.test(message) ? demoItems.filter((item) => item.domain === 'game') : demoItems.filter((item) => item.domain === 'movie')
      const answer = chosen.length ? 'Mình đã chọn vài gợi ý hợp với điều bạn kể. Đây là bản xem thử nên nội dung trò chuyện được mô phỏng.' : 'Cảm ơn bạn đã chia sẻ. Hãy thử kể thêm về tâm trạng hoặc khoảng thời gian bạn có nhé.'
      conversation.messages = [...(conversation.messages || []),
        { id: `gmre-demo-u-${Date.now()}`, role: 'user', content: message, created_at: now },
        { id: `gmre-demo-a-${Date.now()}`, role: 'assistant', content: answer, meta: { items: chosen }, created_at: now },
      ]
      conversation.title = conversation.messages.length <= 2 ? message.slice(0, 56) : conversation.title
      conversation.updated_at = now
      return Promise.resolve({ title: conversation.title, result: { answer, items: chosen } })
    }
    return request<{ title: string; result: { answer: string; items: ApiRecommendation[] } }>(`/conversations/${id}/messages`, token, {
      method: 'POST', body: JSON.stringify({ message, language: 'vi' }),
    })
  },
}
