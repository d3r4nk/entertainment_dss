import { createContext, useContext, useEffect, useMemo, useRef, useState, type CSSProperties, type FormEvent, type PointerEvent as ReactPointerEvent, type ReactNode } from 'react'
import { Link, Navigate, Route, Routes, useLocation, useNavigate, useParams, useSearchParams } from 'react-router-dom'
import {
  ArrowDownLeft, ArrowLeft, ArrowRight, Bookmark, Check, ChevronDown, Clock3, Compass,
  CircleCheck, Heart, Library, LogOut, Menu, MessageCircle, Plus, Search, Send, Settings2,
  ShieldCheck, Sparkles, Star, Trash2, UserRound, X,
} from 'lucide-react'
import { api, ApiError, DEMO_MODE, DEMO_TOKEN, isDemoToken } from './api'
import { catalogue, categories, itemById, recommendationToMedia, starterPrompts, vibes } from './data'
import type { ApiRecommendation, Conversation, MediaItem, TraceRun, User } from './types'

type TasteSettings = { displayName: string; genres: string[]; platforms: string[]; novelty: number; intensity: number; emailTips: boolean; weeklyNote: boolean }
const defaultTaste: TasteSettings = { displayName: '', genres: ['Sci-Fi', 'Tâm lý', 'Indie / Lo-fi'], platforms: ['Netflix', 'Spotify', 'Steam'], novelty: 58, intensity: 42, emailTips: true, weeklyNote: false }
const tokenKey = 'entdec.web.session'
const moodMatches: Record<string, string[]> = {
  'Thư giãn': ['Thư giãn', 'Muốn thả lỏng', 'Lắng đọng'],
  'Tập trung': ['Tập trung'],
  'Muốn cười': ['Muốn cười'],
  'Phiêu lưu': ['Muốn được tiếp sức'],
  'Ở bên nhau': ['Muốn kết nối', 'Muốn cười'],
}

function tasteKey(user: User) { return `entdec.web.taste.${user.id}` }
function readTaste(user?: User | null): TasteSettings {
  if (!user) return defaultTaste
  try { return { ...defaultTaste, ...JSON.parse(localStorage.getItem(tasteKey(user)) || '{}') } } catch { return defaultTaste }
}
function saveKey(user: User) { return `entdec.web.saved.${user.id}` }
function readSaved(user?: User | null): MediaItem[] {
  if (!user) return []
  try { return JSON.parse(localStorage.getItem(saveKey(user)) || '[]') as MediaItem[] } catch { return [] }
}

function App() {
  const [token, setToken] = useState(() => sessionStorage.getItem(tokenKey) || '')
  const [user, setUser] = useState<User | null>(null)
  const [authReady, setAuthReady] = useState(!token)
  const [accountDataReady, setAccountDataReady] = useState(false)
  const [accountSyncEnabled, setAccountSyncEnabled] = useState(false)
  const [menuOpen, setMenuOpen] = useState(false)
  const [taste, setTaste] = useState<TasteSettings>(defaultTaste)
  const [saved, setSaved] = useState<MediaItem[]>([])
  const [completedIds, setCompletedIds] = useState<string[]>([])
  const [ratings, setRatings] = useState<Record<string, 'like' | 'skip'>>({})
  const [pinnedIds, setPinnedIds] = useState<string[]>([])
  const [accountSyncError, setAccountSyncError] = useState('')
  const syncQueue = useRef<Promise<unknown>>(Promise.resolve())
  const syncSnapshot = useRef<Record<string, unknown>>({})
  const location = useLocation()
  const navigate = useNavigate()

  useEffect(() => {
    if (!token) { setUser(null); setAuthReady(true); setAccountDataReady(true); setAccountSyncEnabled(false); return }
    setAuthReady(false)
    setAccountDataReady(false)
    setAccountSyncEnabled(false)
    setAccountSyncError('')
    api.me(token).then((current) => {
      setUser(current)
      if (isDemoToken(token)) {
        setTaste(readTaste(current)); setSaved(readSaved(current))
        try { setCompletedIds(JSON.parse(localStorage.getItem(`entdec.web.completed.${current.id}`) || '[]')) } catch { setCompletedIds([]) }
        try { setPinnedIds(JSON.parse(localStorage.getItem(`entdec.web.pinned.${current.id}`) || '[]')) } catch { setPinnedIds([]) }
        const demoRatings: Record<string, 'like' | 'skip'> = {}
        for (let index = 0; index < localStorage.length; index++) {
          const key = localStorage.key(index)
          if (!key?.startsWith(`entdec.web.rating.${current.id}.`)) continue
          const value = localStorage.getItem(key)
          if (value === 'like' || value === 'skip') demoRatings[key.slice(`entdec.web.rating.${current.id}.`.length)] = value
        }
        setRatings(demoRatings)
        setAccountSyncEnabled(true)
        return
      }
      return api.getAccountState(token).then(({ payload }) => {
        setAccountSyncEnabled(true)
        const hasServerState = Object.keys(payload).length > 0
        if (hasServerState) {
          setTaste({ ...defaultTaste, ...(payload.taste as Partial<TasteSettings> || {}) })
          setSaved(Array.isArray(payload.saved) ? payload.saved as MediaItem[] : [])
          setCompletedIds(Array.isArray(payload.completedIds) ? payload.completedIds as string[] : [])
          setRatings(payload.ratings && typeof payload.ratings === 'object' ? payload.ratings as Record<string, 'like' | 'skip'> : {})
          setPinnedIds(Array.isArray(payload.pinnedIds) ? payload.pinnedIds as string[] : [])
        } else {
          // Import the browser's existing account data once when the server has no state yet.
          setTaste(readTaste(current))
          setSaved(readSaved(current))
          try { setCompletedIds(JSON.parse(localStorage.getItem(`entdec.web.completed.${current.id}`) || '[]')) } catch { setCompletedIds([]) }
          try { setPinnedIds(JSON.parse(localStorage.getItem(`entdec.web.pinned.${current.id}`) || '[]')) } catch { setPinnedIds([]) }
          const importedRatings: Record<string, 'like' | 'skip'> = {}
          for (let index = 0; index < localStorage.length; index++) {
            const key = localStorage.key(index)
            if (!key?.startsWith(`entdec.web.rating.${current.id}.`)) continue
            const value = localStorage.getItem(key)
            if (value === 'like' || value === 'skip') importedRatings[key.slice(`entdec.web.rating.${current.id}.`.length)] = value
          }
          setRatings(importedRatings)
        }
      }).catch((reason: Error) => {
        setAccountSyncEnabled(false)
        setAccountSyncError(`Chưa tải được dữ liệu tài khoản: ${reason.message}`)
        setTaste(readTaste(current)); setSaved(readSaved(current))
        try { setCompletedIds(JSON.parse(localStorage.getItem(`entdec.web.completed.${current.id}`) || '[]')) } catch { setCompletedIds([]) }
        try { setPinnedIds(JSON.parse(localStorage.getItem(`entdec.web.pinned.${current.id}`) || '[]')) } catch { setPinnedIds([]) }
        setRatings({})
      })
    }).catch(() => {
      sessionStorage.removeItem(tokenKey)
      setToken('')
      setUser(null)
    }).finally(() => { setAuthReady(true); setAccountDataReady(true) })
  }, [token])

  useEffect(() => {
    const expireSession = () => { setToken(''); setUser(null); setSaved([]); setTaste(defaultTaste); setCompletedIds([]); setRatings({}); setPinnedIds([]) }
    window.addEventListener('entdec:session-expired', expireSession)
    return () => window.removeEventListener('entdec:session-expired', expireSession)
  }, [])

  useEffect(() => {
    if (!user || !token || !accountDataReady || !accountSyncEnabled) return
    if (isDemoToken(token)) {
      localStorage.setItem(tasteKey(user), JSON.stringify(taste))
      localStorage.setItem(saveKey(user), JSON.stringify(saved))
      localStorage.setItem(`entdec.web.completed.${user.id}`, JSON.stringify(completedIds))
      localStorage.setItem(`entdec.web.pinned.${user.id}`, JSON.stringify(pinnedIds))
      for (const [id, value] of Object.entries(ratings)) localStorage.setItem(`entdec.web.rating.${user.id}.${id}`, value)
      return
    }
    syncSnapshot.current = { taste, saved, completedIds, ratings, pinnedIds }
    const timer = window.setTimeout(() => {
      const payload = { ...syncSnapshot.current }
      syncQueue.current = syncQueue.current.then(() => api.putAccountState(token, payload)).then(() => setAccountSyncError('')).catch((reason: Error) => setAccountSyncError(`Chưa đồng bộ được dữ liệu: ${reason.message}`))
    }, 250)
    return () => window.clearTimeout(timer)
  }, [taste, saved, completedIds, ratings, pinnedIds, user, token, accountDataReady, accountSyncEnabled])

  const signIn = (newToken: string) => {
    sessionStorage.setItem(tokenKey, newToken)
    setAuthReady(false)
    setToken(newToken)
  }
  const signOut = async () => {
    if (token) { try { await api.logout(token) } catch { /* The local session still needs clearing. */ } }
    sessionStorage.removeItem(tokenKey)
    setToken('')
    setUser(null)
    setAccountDataReady(false)
    setSaved([])
    setTaste(defaultTaste)
    setCompletedIds([])
    setRatings({})
    setPinnedIds([])
    navigate('/')
  }
  const toggleSaved = (item: MediaItem) => setSaved((current) => {
    const exists = current.some((savedItem) => savedItem.id === item.id)
    return exists ? current.filter((savedItem) => savedItem.id !== item.id) : [item, ...current]
  })
  const toggleCompleted = (id: string) => setCompletedIds((current) => current.includes(id) ? current.filter((value) => value !== id) : [...current, id])
  const setItemRating = (id: string, rating: 'like' | 'skip' | null) => setRatings((current) => { const next = { ...current }; if (rating) next[id] = rating; else delete next[id]; return next })
  const togglePinned = (id: string) => setPinnedIds((current) => current.includes(id) ? current.filter((value) => value !== id) : [id, ...current])
  const value: AppContextValue = { token, user, authReady, taste, setTaste, saved, toggleSaved, completedIds, toggleCompleted, ratings, setItemRating, pinnedIds, togglePinned, signIn, signOut }

  return <AppContext.Provider value={value}>
    <div className="app-shell">
      <SiteHeader user={user} menuOpen={menuOpen} setMenuOpen={setMenuOpen} onSignOut={signOut} />
      {accountSyncError && <div className="wrap" style={{ paddingTop: 12 }}><InlineError message={accountSyncError}/></div>}
      <main onClick={() => menuOpen && setMenuOpen(false)}>
        {!authReady ? <div className="page-loading"><span className="spinner" /> Đang mở GMRe…</div> : <Routes>
          <Route path="/" element={<Home onSave={toggleSaved} saved={saved} />} />
          <Route path="/login" element={<AuthPage mode="login" onSignIn={signIn} />} />
          <Route path="/register" element={<AuthPage mode="register" onSignIn={signIn} />} />
          <Route path="/explore" element={<Explore onSave={toggleSaved} saved={saved} />} />
          <Route path="/chatbot" element={<PrivateRoute user={user}><ChatPage token={token} /></PrivateRoute>} />
          <Route path="/history" element={<PrivateRoute user={user}><HistoryPage token={token} /></PrivateRoute>} />
          <Route path="/saved" element={<PrivateRoute user={user}><SavedPage items={saved} onSave={toggleSaved} /></PrivateRoute>} />
          <Route path="/account" element={<PrivateRoute user={user}><AccountPage token={token} user={user} taste={taste} setTaste={setTaste} onPasswordChanged={signOut} /></PrivateRoute>} />
          <Route path="/traces" element={<PrivateRoute user={user}><TracePage token={token} /></PrivateRoute>} />
          <Route path="/item/:id" element={<ItemPage onSave={toggleSaved} saved={saved} />} />
          <Route path="*" element={<Navigate to="/" replace />} />
        </Routes>}
      </main>
      <SiteFooter />
      {location.pathname.startsWith('/chatbot') ? null : <a className="floating-ai" href="/chatbot" onClick={(event) => { event.preventDefault(); navigate(user ? '/chatbot' : '/login', user ? undefined : { state: { from: '/chatbot' } }) }} aria-label="Mở trợ lý AI"><Sparkles size={19} /><span>Hỏi AI</span></a>}
    </div>
  </AppContext.Provider>
}

type AppContextValue = {
  token: string; user: User | null; authReady: boolean; taste: TasteSettings; setTaste: (taste: TasteSettings) => void
  saved: MediaItem[]; toggleSaved: (item: MediaItem) => void; completedIds: string[]; toggleCompleted: (id: string) => void
  ratings: Record<string, 'like' | 'skip'>; setItemRating: (id: string, rating: 'like' | 'skip' | null) => void
  pinnedIds: string[]; togglePinned: (id: string) => void; signIn: (token: string) => void; signOut: () => Promise<void>
}
const AppContext = createContext<AppContextValue | null>(null)
function useApp() { const value = useContext(AppContext); if (!value) throw new Error('App context is missing'); return value }

function PrivateRoute({ user, children }: { user: User | null; children: ReactNode }) {
  const location = useLocation()
  if (!user) return <Navigate to="/login" replace state={{ from: location.pathname + location.search }} />
  return <>{children}</>
}

function SiteHeader({ user, menuOpen, setMenuOpen, onSignOut }: { user: User | null; menuOpen: boolean; setMenuOpen: (open: boolean) => void; onSignOut: () => void }) {
  const links = user
    ? [{ to: '/explore', label: 'Khám phá', icon: <Compass size={16} /> }, { to: '/chatbot', label: 'Chatbot AI', icon: <Sparkles size={16} /> }, { to: '/saved', label: 'Đã lưu', icon: <Bookmark size={16} /> }, { to: '/traces', label: 'Trace AI', icon: <Settings2 size={16} /> }]
    : [{ to: '/explore', label: 'Khám phá', icon: <Compass size={16} /> }, { to: '/chatbot', label: 'Chatbot AI', icon: <Sparkles size={16} /> }, { to: '/#collections', label: 'Bộ sưu tập', icon: <Library size={16} /> }]
  return <header className="site-header">
    <div className="nav-inner">
      <Link to="/" className="brand" aria-label="GMRe — Trang chủ"><span className="brand-name">GMRe</span><small className="brand-tagline">KHÁM PHÁ GIẢI TRÍ HỢP GU</small></Link>
      <button className="mobile-menu" aria-label={menuOpen ? 'Đóng điều hướng' : 'Mở điều hướng'} onClick={() => setMenuOpen(!menuOpen)}>{menuOpen ? <X /> : <Menu />}</button>
      <nav className={`main-nav ${menuOpen ? 'open' : ''}`} aria-label="Điều hướng chính">
        <NavLink to="/" end>Trang chủ</NavLink>
        {links.map((link) => <NavLink to={link.to} key={link.to}>{link.label}</NavLink>)}
        {user ? <div className="nav-user">{isDemoToken(sessionStorage.getItem(tokenKey) || '') && <span className="demo-badge">BẢN XEM THỬ</span>}<Link to="/account" className="avatar" aria-label="Tài khoản">{user.email.slice(0, 1).toUpperCase()}</Link><button className="nav-logout" onClick={onSignOut} title="Đăng xuất"><LogOut size={16} /></button></div> : <div className="nav-auth"><Link to="/login" className="nav-login">Đăng nhập</Link><Link to="/register" className="button button-small">Tạo tài khoản <ArrowRight size={14} /></Link></div>}
      </nav>
    </div>
  </header>
}

function NavLink({ to, end = false, children }: { to: string; end?: boolean; children: ReactNode }) {
  const location = useLocation()
  const active = end ? location.pathname === '/' : location.pathname === to || location.pathname.startsWith(`${to}/`)
  return <Link className={`nav-link ${active ? 'active' : ''}`} to={to}>{children}</Link>
}

function SiteFooter() {
  return <footer className="site-footer"><div className="footer-main wrap">
    <div className="footer-brand"><Link to="/" className="brand"><span className="brand-name">GMRe</span><small className="brand-tagline">KHÁM PHÁ GIẢI TRÍ HỢP GU</small></Link><p>Mỗi khoảng rảnh đều xứng đáng với một điều thật hợp gu.</p></div>
    <div><strong>Khám phá</strong><Link to="/explore">Tìm điều mới</Link><Link to="/chatbot">Trò chuyện với AI</Link><Link to="/#collections">Bộ sưu tập</Link></div>
    <div><strong>Không gian của bạn</strong><Link to="/saved">Danh sách đã lưu</Link><Link to="/history">Lịch sử trò chuyện</Link><Link to="/account">Hồ sơ gu giải trí</Link></div>
  </div><div className="footer-bottom wrap"><span>© 2025 GMRe</span><span>Được làm cho những lúc bạn chưa biết chọn gì ✦</span></div></footer>
}

function Home({ onSave, saved }: { onSave: (item: MediaItem) => void; saved: MediaItem[] }) {
  const [activeCategory, setActiveCategory] = useState('all')
  const [activeVibe, setActiveVibe] = useState('all')
  const shown = catalogue.filter((item) => {
    const matchesCategory = activeCategory === 'all' || item.kind === activeCategory
    return matchesCategory && (activeVibe === 'all' || moodMatches[activeVibe].includes(item.mood))
  })
  const navigate = useNavigate()
  const { user } = useApp()
  const openPrompt = (prompt: string) => navigate(user ? `/chatbot?prompt=${encodeURIComponent(prompt)}` : '/login', user ? undefined : { state: { from: `/chatbot?prompt=${encodeURIComponent(prompt)}` } })
  return <>
    <section className="hero-section"><div className="hero-orbit orbit-one" /><div className="hero-orbit orbit-two" />
      <div className="hero-grid wrap"><div className="hero-copy"><div className="eyebrow"><span className="eyebrow-dot" /> TRỢ LÝ GIẢI TRÍ CỦA RIÊNG BẠN</div>
        <h1>Khoảng rảnh<br />của bạn, <em>đúng gu</em><span className="hero-spark">✳</span></h1>
        <p className="hero-lede">Một bộ phim hay một tựa game hợp gu — tụi mình giúp bạn tìm đúng điều đang muốn tận hưởng.</p>
        <div className="hero-actions"><Link to={user ? '/chatbot' : '/login'} state={user ? undefined : { from: '/chatbot' }} className="button button-primary">Trò chuyện với AI <ArrowRight size={17} /></Link><a href="#discover" className="button button-quiet">Khám phá theo tâm trạng <ArrowDownLeft size={17} /></a></div>
        <div className="trust-line"><span>Phim và game được gợi ý theo tâm trạng của bạn.</span></div>
      </div>
      <div className="hero-showcase"><div className="showcase-glow" /><div className="showcase-note note-top"><span className="note-icon">☁</span><span><b>Mood hôm nay</b><small>Chậm lại một chút</small></span><span className="note-spark">✦</span></div>
        <div className="showcase-card"><div className="showcase-cover" style={{ backgroundImage: `linear-gradient(180deg,transparent 45%,rgba(16,17,24,.78)),url(${catalogue[0].image})` }}><span className="cover-label">GỢI Ý RIÊNG CHO BẠN</span><span className="cover-title">Past<br />Lives</span><span className="cover-bottom">Một tối dịu dàng, nhiều suy ngẫm</span></div><div className="showcase-meta"><div><strong>Khớp với tâm trạng</strong><small>Phim · 1 giờ 46 phút</small></div><div className="match-ring">96<small>%</small></div></div></div>
        <div className="showcase-note note-bottom"><span className="ai-dot"><Sparkles size={16} /></span><span><b>GMRe AI</b><small>Đã tìm thấy lựa chọn hợp gu</small></span><span className="live-dot" /></div>
      </div></div>
      <div className="quick-prompt wrap"><div className="quick-prompt-label"><span className="prompt-star">✳</span><span><b>Chưa biết chọn gì?</b><small>Bắt đầu bằng một câu hỏi</small></span></div><div className="prompt-pills">{starterPrompts.slice(0, 3).map((prompt) => <button key={prompt.title} className="prompt-pill" onClick={() => openPrompt(prompt.text)}><span>{prompt.icon}</span>{prompt.title}<ArrowRight size={14} /></button>)}</div></div>
    </section>

    <section id="discover" className="section discover-section"><div className="wrap"><SectionHeading kicker="MỘT CHÚT CẢM HỨNG" title="Hôm nay bạn muốn tận hưởng điều gì?" sub="Chọn một tâm trạng. Phần còn lại, để GMRe lo." />
      <div className="vibe-row"><span className="vibe-label">Mình đang muốn</span>{vibes.map((vibe, index) => <button key={vibe} onClick={() => setActiveVibe(activeVibe === vibe ? 'all' : vibe)} className={`vibe-chip ${activeVibe === vibe ? 'selected' : ''}`}><span>{['☁', '⌁', '☺', '↗', '♡'][index]}</span>{vibe}</button>)}</div>
      <div className="category-tabs">{categories.map((category) => <button disabled={!category.available} className={`${activeCategory === category.id ? 'selected' : ''} ${category.available ? '' : 'coming-soon-tab'}`} onClick={() => category.available && setActiveCategory(category.id)} key={category.id}>{category.icon}<span>{category.label}</span>{!category.available && <small>Sắp ra mắt</small>}</button>)}</div>
      <div className="media-grid">{shown.slice(0, 6).map((item) => <MediaCard key={item.id} item={item} saved={saved.some((entry) => entry.id === item.id)} onSave={onSave} />)}</div>
      <div className="center-cta"><Link to="/explore" className="button button-outline">Xem thêm điều hay <ArrowRight size={16} /></Link><span>Gợi ý được chọn lọc theo tâm trạng của bạn</span></div>
    </div></section>

    <section className="section how-section"><div className="wrap"><SectionHeading kicker="DỄ NHƯ MỘT CUỘC TRÒ CHUYỆN" title="Tìm đúng gu, qua ba bước nhỏ" sub="Không cần biết chính xác mình muốn gì. Chỉ cần kể tụi mình nghe đôi chút." />
      <div className="steps-grid"><Step number="01" icon="☁" title="Chia sẻ tâm trạng" text="Bạn đang vui, mệt hay chỉ có 30 phút rảnh? Cứ kể theo cách tự nhiên nhất."/><Step number="02" icon="✳" title="AI hiểu ngữ cảnh" text="GMRe kết nối tâm trạng, thời gian và những điều bạn thường yêu thích."/><Step number="03" icon="♡" title="Chọn điều hợp gu" text="Nhận gợi ý có lý do rõ ràng — để bạn dễ chọn điều muốn tận hưởng."/></div>
    </div></section>

    <section id="collections" className="conversation-section"><div className="wrap conversation-wrap"><div className="conversation-copy"><div className="eyebrow eyebrow-light"><span className="eyebrow-dot" /> ĐỪNG CHỌN MỘT MÌNH</div><h2>Một cuộc trò chuyện<br />có thể mở ra <em>cả tối vui.</em></h2><p>Nói với AI như nói với một người bạn hiểu gu của bạn. Càng trò chuyện, gợi ý càng gần với điều bạn đang tìm.</p><button className="button button-white" onClick={() => openPrompt('Tối nay mình muốn một bộ phim nhẹ nhàng để xem cùng bạn, khoảng 2 tiếng.')}>Thử trò chuyện ngay <ArrowRight size={16}/></button><div className="conversation-stat"><strong>Không cần biết tên thể loại.</strong><span>Chỉ cần bắt đầu bằng cảm giác của bạn.</span></div></div>
      <div className="chat-preview"><div className="preview-head"><span className="ai-avatar"><Sparkles size={16}/></span><span><b>GMRe AI</b><small><i/> Đang lắng nghe bạn</small></span><span className="preview-more">•••</span></div><div className="preview-messages"><div className="bubble bubble-user">Tối nay mình muốn xem gì đó nhẹ nhàng, nhưng vẫn có chút để nghĩ về.</div><div className="bubble bubble-ai"><span className="bubble-spark">✳</span><div>Mình nghĩ một câu chuyện ấm áp sẽ hợp với tâm trạng của bạn. Đây là lựa chọn đầu tiên nhé:</div><div className="mini-recommend"><div className="mini-cover" style={{ backgroundImage: `url(${catalogue[2].image})` }} /><div><small>PHIM · ĐỜI THƯỜNG</small><strong>Perfect Days</strong><span>Nhẹ nhàng, sâu lắng · 2 giờ 4 phút</span></div><span className="mini-score">91%</span></div><div className="why-note">✦ Vì bạn muốn thư giãn mà vẫn có điều đọng lại.</div></div></div><div className="preview-composer"><span>Kể thêm một chút nhé…</span><button aria-label="Gửi"><ArrowRight size={16}/></button></div></div>
    </div></section>

    <section className="section quote-section"><div className="wrap quote-wrap"><div className="quote-mark">“</div><div className="quote-main"><span className="eyebrow">NHỮNG TRẢI NGHIỆM GMRe DỄ THƯƠNG HƠN</span><blockquote>“Mình thường mất cả buổi chỉ để chọn phim. Giờ thì chỉ cần kể đang có tâm trạng gì — và tối nào cũng tìm được thứ hợp với mình.”</blockquote><div className="quote-person"><span className="quote-avatar">M</span><span><b>Minh Anh</b><small>Người thích những tối thật chậm</small></span><span className="quote-stars">★★★★★</span></div></div><div className="quote-side"><div className="quote-small"><span>🎮</span><p>“Tụi mình tìm được game co-op để chơi cả tối.”</p><small>— Quang, 26 tuổi</small></div><div className="quote-small"><span>🎬</span><p>“Chọn phim nhanh hơn, đúng tâm trạng hơn.”</p><small>— Linh & Nam</small></div></div></div></section>
    <section className="closing-cta"><div className="wrap closing-inner"><div><span className="eyebrow eyebrow-light">TRẢI NGHIỆM GMRe TIẾP THEO ĐANG CHỜ BẠN</span><h2>Để hôm nay có một điều<br/><em>đáng mong chờ.</em></h2></div><Link to={user ? '/chatbot' : '/register'} className="button button-white">Bắt đầu khám phá <ArrowRight size={17}/></Link><span className="closing-star star-one">✳</span><span className="closing-star star-two">✦</span></div></section>
  </>
}

function SectionHeading({ kicker, title, sub }: { kicker: string; title: string; sub: string }) {
  return <div className="section-heading"><span className="eyebrow">{kicker}</span><h2>{title}</h2><p>{sub}</p></div>
}
function Step({ number, icon, title, text }: { number: string; icon: string; title: string; text: string }) {
  return <article className="step-card"><div className="step-top"><span>{number}</span><span className="step-icon">{icon}</span></div><h3>{title}</h3><p>{text}</p></article>
}

function MediaCard({ item, onSave, saved = false, compact = false, completed = false, onComplete }: { item: MediaItem; onSave?: (item: MediaItem) => void; saved?: boolean; compact?: boolean; completed?: boolean; onComplete?: (item: MediaItem) => void }) {
  const { user, toggleSaved, ratings, setItemRating } = useApp()
  const navigate = useNavigate()
  const location = useLocation()
  const rating = ratings[item.id] || null
  const save = () => {
    if (!user) { navigate('/login', { state: { from: location.pathname } }); return }
    ;(onSave || toggleSaved)(item)
  }
  const rate = (next: 'like' | 'skip') => {
    if (!user) { navigate('/login', { state: { from: location.pathname } }); return }
    setItemRating(item.id, next)
  }
  return <article className={`media-card ${compact ? 'compact' : ''}`}>
    <Link to={`/item/${encodeURIComponent(item.id)}`} className="media-image" style={{ backgroundImage: `linear-gradient(180deg,rgba(16,17,24,.02),rgba(16,17,24,.45)),url("${item.image}")` }} aria-label={`Xem ${item.title}`}>
      <span className="media-kind">{item.eyebrow}</span><span className="media-emoji">{item.emoji}</span><span className="match-badge"><Sparkles size={11}/>{item.match}% hợp gu</span>
    </Link>
    <div className="media-content"><div className="media-title-line"><Link to={`/item/${encodeURIComponent(item.id)}`} state={{ item }}><h3>{item.title}</h3></Link><button className={`icon-button bookmark-button ${saved ? 'on' : ''}`} aria-label={saved ? 'Bỏ lưu' : 'Lưu vào danh sách'} onClick={save}><Bookmark size={17} fill={saved ? 'currentColor' : 'none'}/></button></div>
      <div className="media-meta"><span>{item.kind === 'movie' ? 'Phim' : item.kind === 'game' ? 'Game' : item.kind === 'music' ? 'Âm nhạc' : item.kind === 'book' ? 'Sách' : item.kind === 'podcast' ? 'Podcast' : 'Hoạt động'}</span><i/> <span>{item.duration}</span>{item.year && <><i/><span>{item.year}</span></>}</div>
      <p className="media-why">{item.why}</p><div className="media-footer"><div className="tag-list">{item.tags.slice(0, 2).map((tag) => <span key={tag}>{tag}</span>)}</div>{compact && <div className="rating-mini">{onComplete && <button className={completed ? 'completed' : ''} onClick={() => onComplete(item)} aria-label={completed ? 'Đã trải nghiệm' : 'Đánh dấu đã trải nghiệm'} title={completed ? 'Đã trải nghiệm' : 'Đánh dấu đã trải nghiệm'}><CircleCheck size={14}/></button>}<button className={rating === 'like' ? 'liked' : ''} onClick={() => rate('like')} aria-label="Hợp gu" title="Hợp gu"><Heart size={14}/></button><button className={rating === 'skip' ? 'disliked' : ''} onClick={() => rate('skip')} aria-label="Không hợp gu" title="Không hợp gu"><X size={14}/></button></div>}</div>
    </div>
  </article>
}

function Explore({ onSave, saved }: { onSave: (item: MediaItem) => void; saved: MediaItem[] }) {
  const [category, setCategory] = useState('all')
  const [mood, setMood] = useState('Mọi tâm trạng')
  const [query, setQuery] = useState('')
  const filtered = useMemo(() => catalogue.filter((item) => {
    const matchesCategory = category === 'all' || item.kind === category
    const matchesMood = mood === 'Mọi tâm trạng' || (moodMatches[mood] || [mood]).includes(item.mood)
    const matchesSearch = `${item.title} ${item.description} ${item.tags.join(' ')}`.toLocaleLowerCase('vi').includes(query.toLocaleLowerCase('vi'))
    return matchesCategory && matchesMood && matchesSearch
  }), [category, mood, query])
  return <div className="page-wrap wrap"><div className="page-intro"><span className="eyebrow">GÓC KHÁM PHÁ</span><h1>Một điều hay đang<br/><em>chờ bạn tìm thấy.</em></h1><p>Lướt qua những gợi ý được chọn theo tâm trạng, thời gian và cách bạn muốn tận hưởng.</p></div>
    <div className="editorial-banner"><div><span className="eyebrow eyebrow-light">TUYỂN TẬP · NHỮNG NGÀY MƯA</span><h2>Một chút dịu dàng<br/>cho ngày chậm lại.</h2><p>Phim ấm áp và game nhẹ nhàng cho những ngày muốn nghỉ ngơi.</p></div><span className="editorial-art">☂</span><span className="editorial-bubble">Phim & game<br/><b>hợp tâm trạng</b></span></div>
    <div className="explore-toolbar"><div className="search-field"><Search size={17}/><input value={query} onChange={(event) => setQuery(event.target.value)} placeholder="Tìm phim, game, tâm trạng…" aria-label="Tìm nội dung"/>{query && <button onClick={() => setQuery('')} aria-label="Xóa tìm kiếm"><X size={15}/></button>}</div><label className="select-field"><span>Tâm trạng</span><select value={mood} onChange={(event) => setMood(event.target.value)}><option>Mọi tâm trạng</option>{vibes.map((item) => <option key={item}>{item}</option>)}</select><ChevronDown size={15}/></label></div>
    <div className="category-tabs explore-tabs">{categories.map((entry) => <button key={entry.id} disabled={!entry.available} onClick={() => entry.available && setCategory(entry.id)} className={`${category === entry.id ? 'selected' : ''} ${entry.available ? '' : 'coming-soon-tab'}`}>{entry.icon}<span>{entry.label}</span>{!entry.available && <small>Sắp ra mắt</small>}</button>)}</div>
    {filtered.length ? <div className="media-grid">{filtered.map((item) => <MediaCard key={item.id} item={item} onSave={onSave} saved={saved.some((entry) => entry.id === item.id)}/>)}</div> : <EmptyState icon={<Search size={22}/>} title="Chưa thấy điều bạn tìm" text="Thử một từ khóa khác hoặc chọn bộ lọc rộng hơn nhé."/>}
  </div>
}

function ItemPage({ onSave, saved }: { onSave: (item: MediaItem) => void; saved: MediaItem[] }) {
  const { id = '' } = useParams()
  const location = useLocation()
  const { user } = useApp()
  const navigate = useNavigate()
  const decodedId = id
  let cachedItem: MediaItem | undefined
  try { cachedItem = JSON.parse(sessionStorage.getItem(`entdec.web.item.${decodedId}`) || 'null') as MediaItem | undefined } catch { cachedItem = undefined }
  const savedItem = useApp().saved.find((entry) => entry.id === decodedId)
  const item = (location.state as { item?: MediaItem } | null)?.item || cachedItem || savedItem || itemById(decodedId) || catalogue.find((entry) => decodedId.endsWith(String(entry.id)))
  if (!item) return <div className="page-wrap wrap"><EmptyState icon={<Compass/>} title="Chưa tìm thấy gợi ý này" text="Có thể nội dung đã được cập nhật. Mình cùng khám phá lựa chọn khác nhé." action={<Link className="button button-primary" to="/explore">Đi khám phá <ArrowRight size={16}/></Link>}/></div>
  const similar = catalogue.filter((entry) => entry.id !== item.id && (entry.kind === item.kind || entry.mood === item.mood)).slice(0, 3)
  const saveItem = () => { if (!user) { navigate('/login', { state: { from: location.pathname } }); return } onSave(item) }
  return <div className="page-wrap wrap"><Link className="back-link" to="/explore"><ArrowLeft size={16}/> Quay lại khám phá</Link><div className="detail-layout"><div className="detail-art" style={{ backgroundImage: `linear-gradient(180deg,rgba(17,18,22,.02),rgba(17,18,22,.22)),url("${item.image}")` }}><span>{item.emoji}</span></div><div className="detail-copy"><span className="eyebrow">{item.eyebrow}</span><h1>{item.title}</h1><div className="detail-meta"><span>{item.year || item.duration}</span><i/><span>{item.platform || item.mood}</span></div><div className="detail-match"><span className="match-ring">{item.match}<small>%</small></span><span><b>Mức độ hợp gu</b><small>Gợi ý theo tâm trạng bạn chọn</small></span></div><p>{item.description}</p><div className="detail-why"><span>✳</span><div><b>Vì sao gợi ý này hợp với bạn</b><p>{item.why}</p></div></div><div className="tag-list detail-tags">{item.tags.map((tag) => <span key={tag}>{tag}</span>)}</div><button className={`button ${saved.some((entry) => entry.id === item.id) ? 'button-outline' : 'button-primary'}`} onClick={saveItem}><Bookmark size={17}/>{saved.some((entry) => entry.id === item.id) ? 'Đã lưu vào danh sách' : 'Lưu vào danh sách'}</button></div></div><div className="similar-section"><SectionHeading kicker="NẾU BẠN THÍCH ĐIỀU NÀY" title="Có thể bạn cũng sẽ thích" sub="Thêm vài lựa chọn cùng mood để bạn cân nhắc."/><div className="media-grid">{similar.map((entry) => <MediaCard key={entry.id} item={entry}/>)}</div></div></div>
}

function SavedPage({ items, onSave }: { items: MediaItem[]; onSave: (item: MediaItem) => void }) {
  const [tab, setTab] = useState('Tất cả đã lưu')
  const { completedIds, toggleCompleted } = useApp()
  const shown = tab === 'Đã trải nghiệm' ? items.filter((item) => completedIds.includes(item.id)) : tab === 'Muốn xem / nghe / chơi' ? items.filter((item) => !completedIds.includes(item.id)) : items
  return <div className="page-wrap wrap"><div className="page-intro page-intro-row"><div><span className="eyebrow">GÓC RIÊNG CỦA BẠN</span><h1>Danh sách <em>để dành.</em></h1><p>Những điều hay bạn muốn quay lại, khi đúng lúc.</p></div><span className="saved-counter"><Bookmark size={19}/><b>{items.length}</b><small>điều đã lưu</small></span></div><div className="segmented-tabs">{['Tất cả đã lưu', 'Muốn xem / chơi', 'Đã trải nghiệm'].map((item) => <button key={item} className={tab === item ? 'active' : ''} onClick={() => setTab(item)}>{item}</button>)}</div>{shown.length ? <div className="media-grid">{shown.map((item) => <MediaCard key={item.id} item={item} saved onSave={onSave} compact completed={completedIds.includes(item.id)} onComplete={() => toggleCompleted(item.id)}/>)}</div> : <EmptyState icon={<Bookmark size={22}/>} title={tab === 'Đã trải nghiệm' ? 'Chưa có gì ở đây' : 'Danh sách còn đang để trống'} text={tab === 'Đã trải nghiệm' ? 'Khi đã xem hoặc chơi xong, nhấn dấu tích trên thẻ để lưu lại ở đây.' : 'Gặp điều gì hợp gu, nhấn biểu tượng lưu để dành lại cho khoảng rảnh sau nhé.'} action={<Link to="/explore" className="button button-primary">Khám phá gợi ý <ArrowRight size={16}/></Link>}/>}</div>
}

function HistoryPage({ token }: { token: string }) {
  const [search, setSearch] = useState('')
  const [filter, setFilter] = useState('Tất cả')
  const [items, setItems] = useState<Conversation[]>([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState('')
  const { pinnedIds: pinned, togglePinned } = useApp()
  useEffect(() => {
    let mounted = true
    setLoading(true); setError('')
    api.listConversations(token, 0, search).then(async (response) => {
      const detailed = await Promise.all(response.items.map(async (item) => {
        try { return await api.getConversation(token, item.id) } catch { return item }
      }))
      if (mounted) setItems(detailed)
    }).catch((reason: Error) => mounted && setError(reason.message)).finally(() => mounted && setLoading(false))
    return () => { mounted = false }
  }, [token, search])
  const tabs = ['Tất cả', 'Phim', 'Game', 'Đã ghim']
  const filtered = items.filter((item) => {
    if (filter === 'Đã ghim') return pinned.includes(item.id)
    if (filter === 'Phim' || filter === 'Game') return item.messages?.some((message) => (message.meta?.items || []).some((entry) => String(entry.domain || '').toLowerCase() === filter.toLowerCase()))
    return true
  })
  const remove = async (id: string) => { try { await api.deleteConversation(token, id); setItems((current) => current.filter((entry) => entry.id !== id)); if (pinned.includes(id)) togglePinned(id) } catch (reason) { setError((reason as Error).message) } }
  const removeAll = async () => { if (!window.confirm('Xóa toàn bộ lịch sử trò chuyện?')) return; try { await api.deleteAllConversations(token); setItems([]); for (const id of pinned) togglePinned(id) } catch (reason) { setError((reason as Error).message) } }
  const navigate = useNavigate()
  return <div className="page-wrap wrap"><div className="page-intro"><span className="eyebrow">NHỮNG CUỘC TRÒ CHUYỆN CŨ</span><h1>Lịch sử <em>hợp gu.</em></h1><p>Mở lại những lần mình đã cùng nhau tìm thấy điều hay.</p></div><div className="history-tools"><div className="search-field"><Search size={17}/><input value={search} onChange={(event) => setSearch(event.target.value)} placeholder="Tìm cuộc trò chuyện…" aria-label="Tìm lịch sử"/>{search && <button onClick={() => setSearch('')} aria-label="Xóa tìm kiếm"><X size={15}/></button>}</div><div className="filter-pills">{tabs.map((tab) => <button className={filter === tab ? 'active' : ''} onClick={() => setFilter(tab)} key={tab}>{tab === 'Đã ghim' && <Star size={13}/>} {tab}</button>)}<button onClick={() => void removeAll()}>Xóa tất cả</button></div></div>
    {error && <InlineError message={error}/>} {loading ? <div className="page-loading"><span className="spinner"/> Đang tìm lại những cuộc trò chuyện…</div> : filtered.length ? <div className="history-list">{filtered.map((conversation) => <article className="history-card" key={conversation.id}><div className="history-icon"><MessageCircle size={20}/></div><div className="history-main"><Link to={`/chatbot?conversation=${conversation.id}`}><h3>{conversation.title}</h3></Link><div className="history-meta"><span><Clock3 size={13}/>{formatDate(conversation.updated_at)}</span><i/><span>{conversation.messages?.length || 0} tin nhắn</span><i/><span>{historyKind(conversation)}</span></div><div className="history-preview">{(conversation.messages || []).filter((message) => message.role === 'assistant').flatMap((message) => message.meta?.items || []).slice(0, 3).map((item, index) => <span className="history-thumb" key={`${String(item.id || item.title)}-${index}`}>{String(item.title || 'Gợi ý').slice(0, 20)}</span>)}</div></div><div className="history-actions"><button onClick={() => togglePinned(conversation.id)} className={pinned.includes(conversation.id) ? 'pinned' : ''} title={pinned.includes(conversation.id) ? 'Bỏ ghim' : 'Ghim'}><Star size={16} fill={pinned.includes(conversation.id) ? 'currentColor' : 'none'}/></button><button onClick={() => navigate(`/chatbot?conversation=${conversation.id}`)} title="Tiếp tục trò chuyện"><ArrowRight size={17}/></button><button onClick={() => remove(conversation.id)} title="Xóa"><Trash2 size={16}/></button></div></article>)}</div> : <EmptyState icon={<MessageCircle size={22}/>} title={search ? 'Chưa tìm thấy cuộc trò chuyện' : 'Chưa có lịch sử trò chuyện'} text={search ? 'Thử một từ khóa khác nhé.' : 'Mỗi lần trò chuyện sẽ được lưu tại đây để bạn dễ dàng quay lại.'} action={<Link className="button button-primary" to="/chatbot">Bắt đầu trò chuyện <ArrowRight size={16}/></Link>}/>}</div>
}

function historyKind(conversation: Conversation) {
  const kinds = new Set((conversation.messages || []).flatMap((message) => (message.meta?.items || []).map((entry) => String(entry.domain || ''))))
  if (kinds.has('movie')) return 'Phim & series'
  if (kinds.has('game')) return 'Game'
  return 'Gợi ý giải trí'
}

function TracePage({ token }: { token: string }) {
  const [status, setStatus] = useState('')
  const [domain, setDomain] = useState('')
  const [items, setItems] = useState<Record<string, unknown>[]>([])
  const [stats, setStats] = useState<Record<string, unknown>>({})
  const [selected, setSelected] = useState<Record<string, unknown> | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState('')
  useEffect(() => {
    let mounted = true
    setLoading(true); setError('')
    const filters = { ...(status ? { status } : {}), ...(domain ? { domain } : {}) }
    Promise.all([api.listTraces(token, filters), api.traceStats(token, filters)]).then(([list, summary]) => {
      if (mounted) { setItems(list.items); setStats(summary) }
    }).catch((reason: Error) => mounted && setError(reason.message)).finally(() => mounted && setLoading(false))
    return () => { mounted = false }
  }, [token, status, domain])
  const openTrace = async (id: string) => {
    setError('')
    try {
      const trace = await api.getTrace(token, id)
      const events = await api.getTraceEvents(token, id)
      setSelected({ ...trace, events })
    }
    catch (reason) { setError((reason as Error).message) }
  }
  const count = typeof stats.count === 'number' ? stats.count : items.length
  return <div className="page-wrap wrap"><div className="page-intro"><span className="eyebrow">CÔNG CỤ THEO DÕI</span><h1>Trace <em>trợ lý.</em></h1><p>Xem trạng thái xử lý và thời gian của các lượt gợi ý thuộc tài khoản này.</p></div>
    <div className="account-overview"><div className="account-stat"><strong>{count}</strong><span>lượt xử lý</span></div><div className="account-stat"><strong>{Object.keys((stats.statuses as object) || {}).length}</strong><span>trạng thái ghi nhận</span></div></div>
    <div className="history-tools"><label className="select-field"><span>Trạng thái</span><select value={status} onChange={(event) => setStatus(event.target.value)}><option value="">Tất cả</option>{['queued','running','complete','partial','error','interrupted'].map((item) => <option key={item}>{item}</option>)}</select><ChevronDown size={15}/></label><label className="select-field"><span>Danh mục</span><select value={domain} onChange={(event) => setDomain(event.target.value)}><option value="">Phim và game</option><option value="movie">Phim</option><option value="game">Game</option></select><ChevronDown size={15}/></label></div>
    {error && <InlineError message={error}/>} {loading ? <div className="page-loading"><span className="spinner"/> Đang tải trace…</div> : items.length ? <div className="history-list">{items.map((item, index) => { const id = String(item.trace_id || item.id || index); return <article className="history-card" key={id}><div className="history-icon"><Settings2 size={19}/></div><div className="history-main"><button className="chat-rec-title" onClick={() => void openTrace(id)}><b>{String(item.question || 'Lượt gợi ý')}</b></button><div className="history-meta"><span>{String(item.status || '—')}</span><i/><span>{String(item.profile || 'profile chưa ghi')}</span><i/><span>{item.created_at ? formatDate(String(item.created_at)) : '—'}</span></div></div><button className="button button-outline" onClick={() => void openTrace(id)}>Chi tiết</button></article> })}</div> : <EmptyState icon={<Settings2 size={22}/>} title="Chưa có trace" text="Trace sẽ xuất hiện sau khi backend ghi nhận lượt xử lý AI."/>}
    {selected && <section className="settings-card" style={{ marginTop: 24 }}><div className="settings-title"><div><h2>Chi tiết trace</h2><p>{String(selected.trace_id || '')} · {String(selected.status || '')}</p></div><button className="icon-button" onClick={() => setSelected(null)} aria-label="Đóng chi tiết"><X size={17}/></button></div><pre className="trace-json">{JSON.stringify(selected, null, 2)}</pre></section>}
  </div>
}
function formatDate(date: string) { try { return new Intl.DateTimeFormat('vi-VN', { day: 'numeric', month: 'short', hour: '2-digit', minute: '2-digit' }).format(new Date(date)) } catch { return 'Vừa trò chuyện' } }

function AccountPage({ token, user, taste, setTaste, onPasswordChanged }: { token: string; user: User | null; taste: TasteSettings; setTaste: (next: TasteSettings) => void; onPasswordChanged: () => void }) {
  const [passwords, setPasswords] = useState({ current_password: '', new_password: '', confirm_password: '' })
  const [feedback, setFeedback] = useState('')
  const [error, setError] = useState('')
  const [busy, setBusy] = useState(false)
  const [sessionCount, setSessionCount] = useState(0)
  const { saved } = useApp()
  useEffect(() => { api.listConversations(token).then((result) => setSessionCount(result.total)).catch(() => setSessionCount(0)) }, [token])
  const genres = ['Sci-Fi', 'Indie / Lo-fi', 'RPG', 'Tâm lý', 'Sách kỹ năng', 'Cà phê & triển lãm', 'Hài hước', 'Phiêu lưu']
  const platforms = ['Netflix', 'Spotify', 'Steam', 'YouTube', 'Apple Books', 'MUBI']
  const toggleValue = (key: 'genres' | 'platforms', value: string) => setTaste({ ...taste, [key]: taste[key].includes(value) ? taste[key].filter((entry) => entry !== value) : [...taste[key], value] })
  const submitPassword = async (event: FormEvent) => {
    event.preventDefault(); setError(''); setFeedback('')
    if (passwords.new_password !== passwords.confirm_password) { setError('Hai mật khẩu mới chưa trùng khớp.'); return }
    setBusy(true)
    try { await api.changePassword(token, passwords); setFeedback('Đổi mật khẩu thành công. Bạn sẽ cần đăng nhập lại.'); setTimeout(onPasswordChanged, 1200) }
    catch (reason) { setError((reason as Error).message) }
    finally { setBusy(false) }
  }
  const update = (patch: Partial<TasteSettings>) => setTaste({ ...taste, ...patch })
  return <div className="page-wrap wrap account-page"><div className="page-intro"><span className="eyebrow">GÓC CỦA RIÊNG BẠN</span><h1>Gu của bạn, <em>qua từng ngày.</em></h1><p>Điều chỉnh sở thích để những gợi ý ngày càng gần với mình hơn.</p></div>
    <div className="account-overview"><div className="account-avatar">{(taste.displayName || user?.email || 'K').slice(0,1).toUpperCase()}</div><div className="account-identity"><span className="eyebrow">THÀNH VIÊN TRẢI NGHIỆM GMRe</span><h2>{taste.displayName || user?.email.split('@')[0]}</h2><p>{user?.email}</p></div><div className="account-stat"><strong>{sessionCount}</strong><span>phiên gợi ý</span></div><div className="account-stat"><strong>{saved.length}</strong><span>mục đã lưu</span></div><div className="account-stat"><strong>{taste.genres[0] || 'Đang khám phá'}</strong><span>gu nổi bật</span></div></div>
    <div className="account-grid"><section className="settings-card taste-card"><div className="settings-title"><div className="settings-icon purple"><Sparkles size={18}/></div><div><h2>DNA giải trí</h2><p>Chọn những điều thường khiến bạn vui.</p></div></div><div className="setting-block"><label className="setting-label">Thể loại và gu yêu thích</label><div className="choice-chips">{genres.map((genre) => <button key={genre} className={taste.genres.includes(genre) ? 'selected' : ''} onClick={() => toggleValue('genres', genre)}>{taste.genres.includes(genre) && <Check size={13}/>} {genre}</button>)}</div></div><div className="setting-block"><label className="setting-label">Nền tảng bạn hay dùng</label><div className="choice-chips platform-chips">{platforms.map((platform) => <button key={platform} className={taste.platforms.includes(platform) ? 'selected' : ''} onClick={() => toggleValue('platforms', platform)}>{taste.platforms.includes(platform) && <Check size={13}/>} {platform}</button>)}</div></div><RangeSetting label="Quen thuộc" left="Ở vùng an toàn" right="Khám phá mới lạ" value={taste.novelty} onChange={(novelty) => update({ novelty })}/><RangeSetting label="Cường độ" left="Nhẹ nhàng thư giãn" right="Kịch tính cao" value={taste.intensity} onChange={(intensity) => update({ intensity })}/></section>
      <div className="account-side"><section className="settings-card"><div className="settings-title"><div className="settings-icon"><UserRound size={18}/></div><div><h2>Thông tin cá nhân</h2><p>Mình nên gọi bạn là gì?</p></div></div><label className="form-label">Tên hiển thị<input className="text-input" value={taste.displayName} onChange={(event) => update({ displayName: event.target.value })} placeholder={user?.email.split('@')[0] || 'Tên của bạn'}/></label><label className="form-label">Địa chỉ email<input className="text-input disabled-input" value={user?.email || ''} readOnly/></label></section><section className="settings-card"><div className="settings-title"><div className="settings-icon"><Settings2 size={18}/></div><div><h2>Điều bạn muốn nhận</h2><p>Tuỳ chỉnh nhắc nhở từ GMRe.</p></div></div><ToggleSetting title="Gợi ý mới hợp gu" description="Khi có bộ sưu tập mới dành cho bạn" checked={taste.emailTips} onChange={(emailTips) => update({ emailTips })}/><ToggleSetting title="Bản tin GMRe" description="Một email nhẹ nhàng mỗi tuần" checked={taste.weeklyNote} onChange={(weeklyNote) => update({ weeklyNote })}/></section></div>
      <section className="settings-card password-card"><div className="settings-title"><div className="settings-icon coral"><ShieldCheck size={18}/></div><div><h2>Bảo mật tài khoản</h2><p>Đổi mật khẩu sẽ đăng xuất tài khoản khỏi các phiên hiện có.</p></div></div>{isDemoToken(token) && <div className="demo-note">Bạn đang xem bản demo. Thông tin tài khoản và đổi mật khẩu chỉ hoạt động khi kết nối API thật.</div>}<form className="password-form" onSubmit={submitPassword}><label className="form-label">Mật khẩu hiện tại<input className="text-input" type="password" autoComplete="current-password" minLength={10} value={passwords.current_password} onChange={(event) => setPasswords({ ...passwords, current_password: event.target.value })} required/></label><label className="form-label">Mật khẩu mới<input className="text-input" type="password" autoComplete="new-password" minLength={10} value={passwords.new_password} onChange={(event) => setPasswords({ ...passwords, new_password: event.target.value })} required/><small>Dùng ít nhất 10 ký tự.</small></label><label className="form-label">Nhập lại mật khẩu mới<input className="text-input" type="password" autoComplete="new-password" minLength={10} value={passwords.confirm_password} onChange={(event) => setPasswords({ ...passwords, confirm_password: event.target.value })} required/></label><div className="password-submit">{error && <InlineError message={error}/>} {feedback && <div className="success-note"><Check size={15}/>{feedback}</div>}<button className="button button-outline" disabled={busy || isDemoToken(token)}>{busy ? 'Đang cập nhật…' : 'Cập nhật mật khẩu'}</button></div></form></section>
    </div></div>
}
function RangeSetting({ label, left, right, value, onChange }: { label: string; left: string; right: string; value: number; onChange: (value: number) => void }) { return <div className="range-setting"><div className="range-head"><span>{label}</span><small>{value}%</small></div><input type="range" min="0" max="100" value={value} onChange={(event) => onChange(Number(event.target.value))}/><div className="range-labels"><span>{left}</span><span>{right}</span></div></div> }
function ToggleSetting({ title, description, checked, onChange }: { title: string; description: string; checked: boolean; onChange: (value: boolean) => void }) { return <label className="toggle-setting"><span><b>{title}</b><small>{description}</small></span><input type="checkbox" checked={checked} onChange={(event) => onChange(event.target.checked)}/><i/></label> }

function ChatPage({ token }: { token: string }) {
  const [params, setParams] = useSearchParams()
  const conversationId = params.get('conversation')
  const [threads, setThreads] = useState<Conversation[]>([])
  const [conversation, setConversation] = useState<Conversation | null>(null)
  const [draft, setDraft] = useState('')
  const [busy, setBusy] = useState(false)
  const [pendingQuestion, setPendingQuestion] = useState('')
  const [activeTraceId, setActiveTraceId] = useState<string | null>(null)
  const [openTraceId, setOpenTraceId] = useState<string | null>(null)
  const [closedTraceId, setClosedTraceId] = useState<string | null>(null)
  const [traceRuns, setTraceRuns] = useState<Record<string, TraceRun>>({})
  const [traceErrors, setTraceErrors] = useState<Record<string, string>>({})
  const [paneRatio, setPaneRatio] = useState(() => {
    try { return Number(sessionStorage.getItem('gmre.chat-layout.ratio')) || 0.4 } catch { return 0.4 }
  })
  const [paneHeights, setPaneHeights] = useState(() => {
    try {
      const saved = JSON.parse(sessionStorage.getItem('gmre.chat-layout.heights') || '{}')
      return { chat: Number(saved.chat) || 460, trace: Number(saved.trace) || 460 }
    } catch { return { chat: 460, trace: 460 } }
  })
  const [areaWidth, setAreaWidth] = useState(0)
  const [error, setError] = useState('')
  const [sidebarOpen, setSidebarOpen] = useState(true)
  const [context, setContext] = useState<string[]>([])
  const bottomRef = useRef<HTMLDivElement | null>(null)
  const scrollRef = useRef<HTMLDivElement | null>(null)
  const followChatRef = useRef(true)
  const workAreaRef = useRef<HTMLDivElement | null>(null)
  const widthDragRef = useRef<{ startX: number; startRatio: number } | null>(null)
  const heightDragRef = useRef<{ pane: 'chat' | 'trace'; startY: number; startHeight: number } | null>(null)
  const starters = starterPrompts

  const loadThreads = async () => {
    try { setThreads((await api.listConversations(token, 0)).items) } catch (reason) { setError((reason as Error).message) }
  }
  useEffect(() => { void loadThreads() }, [token])
  useEffect(() => {
    let mounted = true
    setError('')
    if (conversationId) {
      api.getConversation(token, conversationId).then((result) => mounted && setConversation(result)).catch((reason: Error) => mounted && setError(reason.message))
    } else setConversation(null)
    return () => { mounted = false }
  }, [token, conversationId])
  useEffect(() => { const prompt = params.get('prompt'); if (prompt) { setDraft(prompt); setParams((previous) => { previous.delete('prompt'); return previous }, { replace: true }) } }, [params, setParams])
  useEffect(() => {
    try { sessionStorage.setItem('gmre.chat-layout.ratio', String(paneRatio)) } catch { /* Optional per-tab preference. */ }
  }, [paneRatio])
  useEffect(() => {
    try { sessionStorage.setItem('gmre.chat-layout.heights', JSON.stringify(paneHeights)) } catch { /* Optional per-tab preference. */ }
  }, [paneHeights])
  useEffect(() => {
    const area = workAreaRef.current
    if (!area) return
    const measure = () => setAreaWidth(area.clientWidth)
    measure()
    const observer = new ResizeObserver(measure)
    observer.observe(area)
    return () => observer.disconnect()
  }, [sidebarOpen, Boolean(openTraceId)])
  useEffect(() => {
    const traceIds = [...new Set([activeTraceId, openTraceId].filter((id): id is string => Boolean(id)))]
    if (!traceIds.length) return
    let stopped = false
    let timer: number | undefined
    const poll = async () => {
      let hasActive = false
      for (const id of traceIds) {
        try {
          const run = await api.getTrace(token, id)
          if (stopped) return
          setTraceRuns((current) => ({ ...current, [id]: run }))
          setTraceErrors((current) => { const next = { ...current }; delete next[id]; return next })
          if (run.status === 'queued' || run.status === 'running') hasActive = true
          if (id === activeTraceId && !['queued', 'running'].includes(run.status)) {
            setBusy(false)
            setActiveTraceId(null)
            setPendingQuestion('')
            if (run.status === 'error' || run.status === 'interrupted') {
              setError(run.error || 'Lượt gợi ý chưa hoàn tất. Bạn có thể thử gửi lại câu hỏi.')
            } else if (conversationId) {
              try {
                const updated = await api.getConversation(token, conversationId)
                if (!stopped) setConversation(updated)
                void loadThreads()
              } catch (reason) { if (!stopped) setError((reason as Error).message) }
            }
          }
        } catch (reason) {
          if (!stopped) setTraceErrors((current) => ({ ...current, [id]: (reason as Error).message }))
          if (id === activeTraceId) hasActive = true
        }
      }
      if (!stopped && hasActive) timer = window.setTimeout(poll, 1000)
    }
    void poll()
    return () => { stopped = true; if (timer) window.clearTimeout(timer) }
  }, [token, activeTraceId, openTraceId, conversationId])
  useEffect(() => {
    if (followChatRef.current) bottomRef.current?.scrollIntoView({ behavior: 'smooth', block: 'end' })
  }, [conversation?.messages?.length, busy, pendingQuestion, traceRuns[activeTraceId || '']?.answer])

  const selectConversation = (id?: string) => {
    setPendingQuestion(''); setActiveTraceId(null); setBusy(false); setOpenTraceId(null); setClosedTraceId(null)
    setParams(id ? { conversation: id } : {}, { replace: false })
    setError('')
  }
  const send = async (event?: FormEvent) => {
    event?.preventDefault()
    const clean = draft.trim()
    if (!clean || busy) return
    setDraft(''); setBusy(true); setPendingQuestion(clean); setError('')
    try {
      const active = conversation || await api.createConversation(token, clean.slice(0, 56))
      if (!conversation) {
        setConversation({ ...active, messages: [] })
        setParams({ conversation: active.id }, { replace: false })
      }
      const submitted = await api.submitConversationRun(token, active.id, clean, 'vi')
      setActiveTraceId(submitted.trace_id)
      setTraceRuns((current) => ({
        ...current,
        [submitted.trace_id]: {
          trace_id: submitted.trace_id,
          conversation_id: active.id,
          question: clean,
          status: 'queued',
          events: [],
        },
      }))
      setClosedTraceId(null)
      followChatRef.current = true
    } catch (reason) {
      setDraft(clean); setPendingQuestion(''); setBusy(false); setError((reason as Error).message)
    }
  }
  const attachContext = (value: string) => {
    setContext((current) => current.includes(value) ? current.filter((entry) => entry !== value) : [...current, value])
    if (!draft.includes(value)) setDraft((current) => `${current}${current ? ' · ' : ''}${value}`)
  }
  const deleteThread = async (id: string) => {
    try { await api.deleteConversation(token, id); setThreads((current) => current.filter((item) => item.id !== id)); if (conversationId === id) selectConversation() }
    catch (reason) { setError((reason as Error).message) }
  }
  const messages = conversation?.messages || []
  const selectedTraceId = openTraceId || ''
  const selectedTrace = selectedTraceId ? traceRuns[selectedTraceId] : undefined
  const ratio = Math.max(0.32, Math.min(0.68, paneRatio))
  const maxHeight = Math.max(320, window.innerHeight - 150)
  const usableWidth = Math.max(1, areaWidth - 54)
  const paneStyle = {
    '--chat-pane-width': `${Math.max(320, usableWidth * ratio)}px`,
  } as CSSProperties
  const chatPaneHeight = `${Math.max(320, Math.min(maxHeight, paneHeights.chat))}px`
  const tracePaneHeight = `${Math.max(320, Math.min(maxHeight, paneHeights.trace))}px`
  const beginWidthResize = (event: ReactPointerEvent<HTMLDivElement>) => {
    if (areaWidth < 700) return
    widthDragRef.current = { startX: event.clientX, startRatio: ratio }
    event.currentTarget.setPointerCapture(event.pointerId)
    event.preventDefault()
  }
  const moveWidthResize = (event: ReactPointerEvent<HTMLDivElement>) => {
    const drag = widthDragRef.current
    if (!drag || !workAreaRef.current) return
    const available = workAreaRef.current.clientWidth - 54
    setPaneRatio(Math.max(320 / available, Math.min(1 - 320 / available, drag.startRatio + (event.clientX - drag.startX) / available)))
  }
  const beginHeightResize = (pane: 'chat' | 'trace', event: ReactPointerEvent<HTMLDivElement>) => {
    heightDragRef.current = { pane, startY: event.clientY, startHeight: paneHeights[pane] }
    event.currentTarget.setPointerCapture(event.pointerId)
    event.preventDefault()
  }
  const moveHeightResize = (event: ReactPointerEvent<HTMLDivElement>) => {
    const drag = heightDragRef.current
    if (!drag) return
    setPaneHeights((current) => ({ ...current, [drag.pane]: Math.max(320, Math.min(maxHeight, drag.startHeight + event.clientY - drag.startY)) }))
  }
  const openTrace = (id: string) => { setOpenTraceId(id); setClosedTraceId(null) }
  const closeTrace = () => { setClosedTraceId(openTraceId); setOpenTraceId(null) }
  const reopenTrace = () => { if (closedTraceId) openTrace(closedTraceId) }
  return <div className={`chat-workspace ${sidebarOpen ? '' : 'sidebar-collapsed'}`}>
    <aside className="chat-sidebar"><div className="chat-sidebar-top"><div className="chat-sidebar-title"><span className="ai-avatar"><Sparkles size={16}/></span><span><b>GMRe</b><small>Không gian trò chuyện</small></span></div><button className="icon-button sidebar-toggle" onClick={() => setSidebarOpen(false)} aria-label="Thu gọn danh sách"><ArrowLeft size={16}/></button></div><button className="new-chat-button" onClick={() => selectConversation()}><Plus size={16}/> Cuộc trò chuyện mới</button><div className="sidebar-section-label">GẦN ĐÂY <span>{threads.length}</span></div><div className="thread-list">{threads.map((thread) => <div className={`thread-entry ${conversationId === thread.id ? 'current' : ''}`} key={thread.id}><button onClick={() => selectConversation(thread.id)}><MessageCircle size={15}/><span>{thread.title}</span></button><button className="thread-delete" title="Xóa trò chuyện" onClick={() => deleteThread(thread.id)}><Trash2 size={14}/></button></div>)}{!threads.length && <p className="sidebar-empty">Những cuộc trò chuyện mới sẽ xuất hiện ở đây.</p>}</div><div className="sidebar-note"><span>✳</span><p>Gợi ý càng hợp hơn khi bạn kể thêm về tâm trạng và khoảng thời gian mình có.</p></div><Link to="/history" className="sidebar-history"><Clock3 size={15}/> Xem toàn bộ lịch sử <ArrowRight size={14}/></Link></aside>
    <section className="chat-main"><div className="chat-topbar"><button className="icon-button open-sidebar" onClick={() => setSidebarOpen(true)} aria-label="Mở danh sách trò chuyện"><Menu size={18}/></button><div><span className="chat-status-dot"/> Trợ lý AI đang sẵn sàng</div><Link to="/account" className="chat-profile"><Settings2 size={16}/> Gu của bạn</Link></div>
      {error && <div className="chat-error"><InlineError message={error}/><button onClick={() => setError('')} aria-label="Đóng thông báo"><X size={16}/></button></div>}
      <div ref={workAreaRef} className={`chat-workarea ${openTraceId ? 'has-trace' : ''} ${areaWidth >= 700 ? 'wide-trace' : ''}`} style={paneStyle}>
        <section className="chat-pane" style={{ height: chatPaneHeight }}>
          {!openTraceId && closedTraceId && <button className="reopen-trace-button" onClick={reopenTrace}><Settings2 size={14}/> Mở lại chi tiết</button>}
          <div ref={scrollRef} className={`chat-scroll ${messages.length || pendingQuestion ? 'has-messages' : ''}`} onScroll={(event) => {
            const element = event.currentTarget
            followChatRef.current = element.scrollHeight - element.clientHeight - element.scrollTop <= 48
          }}>
            {!messages.length && !pendingQuestion ? <div className="chat-welcome"><span className="welcome-spark">✳</span><span className="eyebrow">TRẢI NGHIỆM GMRe AI</span><h1>Chào bạn,<br/><em>mình cùng tìm nhé.</em></h1><p>Hôm nay bạn đang có tâm trạng thế nào? Cứ kể tự nhiên — dù bạn chỉ biết mình muốn cảm thấy điều gì.</p><div className="starter-grid">{starters.map((prompt) => <button key={prompt.title} onClick={() => setDraft(prompt.text)}><span>{prompt.icon}</span><b>{prompt.title}</b><small>{prompt.text}</small><ArrowRight size={15}/></button>)}</div><div className="prompt-context"><span>Gợi ý bắt đầu:</span>{['Thư giãn', 'Dưới 2 tiếng', 'Đi cùng bạn bè'].map((item) => <button className={context.includes(item) ? 'selected' : ''} onClick={() => attachContext(item)} key={item}>{item}</button>)}</div></div> : <div className="message-stream">{messages.map((message, index) => {
              const next = messages[index + 1]
              const traceId = message.role === 'user' && next?.role === 'assistant' ? String(next.meta?.trace_id || '') : ''
              return <ChatMessage key={message.id} message={message} traceId={traceId} onOpenTrace={openTrace}/>
            })}
              {pendingQuestion && <ChatMessage message={{ id: `pending-${activeTraceId || 'question'}`, role: 'user', content: pendingQuestion }} traceId={activeTraceId || ''} onOpenTrace={openTrace}/>}
              {busy && <div className="typing-row"><span className="ai-avatar"><Sparkles size={14}/></span><div><span className="typing-dots"><i/><i/><i/></span><small>Đang tìm điều hợp gu bạn…</small></div></div>}
              <div ref={bottomRef}/>
            </div>}
          </div>
          <div className="composer-wrap"><div className="composer-context">{['☁ Tâm trạng', '◷ Thời lượng', '♡ Đi cùng ai'].map((item) => <button key={item} onClick={() => attachContext(item.split(' ').slice(1).join(' '))}>{item}</button>)}</div><form className="chat-composer" onSubmit={send}><textarea rows={1} value={draft} onChange={(event) => setDraft(event.target.value)} onKeyDown={(event) => { if (event.key === 'Enter' && !event.shiftKey) { event.preventDefault(); void send() } }} placeholder="Mình có thể giúp bạn tìm điều gì?" aria-label="Tin nhắn cho AI"/><button className="send-button" disabled={!draft.trim() || busy} aria-label="Gửi tin nhắn">{busy ? <span className="spinner spinner-white"/> : <Send size={17}/>}</button></form><small className="composer-hint">Enter để gửi · Shift + Enter để xuống dòng · AI tập trung vào gợi ý phim và game</small></div>
          <div className="pane-height-handle" role="separator" tabIndex={0} aria-label="Đổi chiều cao khung chat" aria-orientation="horizontal" onPointerDown={(event) => beginHeightResize('chat', event)} onPointerMove={moveHeightResize} onPointerUp={() => { heightDragRef.current = null }} onPointerCancel={() => { heightDragRef.current = null }} onKeyDown={(event) => { if (event.key === 'ArrowUp' || event.key === 'ArrowDown') { event.preventDefault(); setPaneHeights((current) => ({ ...current, chat: Math.max(320, Math.min(maxHeight, current.chat + (event.key === 'ArrowUp' ? -16 : 16))) })) } }}/>
        </section>
        {openTraceId && <>
          <div className="pane-width-handle" role="separator" tabIndex={0} aria-label="Đổi chiều rộng khung chat và truy vấn" aria-orientation="vertical" aria-valuenow={Math.round(ratio * 100)} onPointerDown={beginWidthResize} onPointerMove={moveWidthResize} onPointerUp={() => { widthDragRef.current = null }} onPointerCancel={() => { widthDragRef.current = null }} onKeyDown={(event) => { if (event.key === 'ArrowLeft' || event.key === 'ArrowRight') { event.preventDefault(); setPaneRatio((current) => Math.max(0.32, Math.min(0.68, current + (event.key === 'ArrowLeft' ? -0.02 : 0.02)))) } }}/>
          <section className="trace-pane" style={{ height: tracePaneHeight }}>
            <div className="trace-pane-header"><div><span className="eyebrow">THEO DÕI TRỰC TIẾP</span><h2>Chi tiết truy vấn</h2></div><button className="button button-outline trace-close" onClick={closeTrace}>Đóng chi tiết</button></div>
            <div className="trace-scroll">
              {traceErrors[openTraceId] && <InlineError message={traceErrors[openTraceId]}/>}
              {!selectedTrace ? <div className="trace-loading"><span className="spinner"/> Đang tải chi tiết truy vấn…</div> : <TraceDetails run={selectedTrace}/>}
            </div>
            <div className="pane-height-handle" role="separator" tabIndex={0} aria-label="Đổi chiều cao khung truy vấn" aria-orientation="horizontal" onPointerDown={(event) => beginHeightResize('trace', event)} onPointerMove={moveHeightResize} onPointerUp={() => { heightDragRef.current = null }} onPointerCancel={() => { heightDragRef.current = null }} onKeyDown={(event) => { if (event.key === 'ArrowUp' || event.key === 'ArrowDown') { event.preventDefault(); setPaneHeights((current) => ({ ...current, trace: Math.max(320, Math.min(maxHeight, current.trace + (event.key === 'ArrowUp' ? -16 : 16))) })) }}/>
          </section>
        </>}
      </div>
    </section>
  </div>
}

function ChatMessage({ message, traceId, onOpenTrace }: {
  message: { id: string; role: 'user' | 'assistant'; content: string; meta?: { items?: ApiRecommendation[] } | null }
  traceId?: string
  onOpenTrace?: (id: string) => void
}) {
  const isUser = message.role === 'user'
  const items = (message.meta?.items || []).map((item, index) => recommendationToMedia(item as Record<string, unknown>, index))
  const { toggleSaved, saved } = useApp()
  const navigate = useNavigate()
  const openItem = (item: MediaItem) => {
    sessionStorage.setItem(`entdec.web.item.${item.id}`, JSON.stringify(item))
    navigate(`/item/${encodeURIComponent(item.id)}`, { state: { item } })
  }
  return <div className={`chat-message ${isUser ? 'from-user' : 'from-ai'}`}>{!isUser && <span className="ai-avatar"><Sparkles size={15}/></span>}<div className="message-body"><div className="message-bubble">{message.content}</div>{isUser && traceId && onOpenTrace && <button className="message-trace-button" onClick={() => onOpenTrace(traceId)}><Settings2 size={13}/> Mở rộng chi tiết truy vấn</button>}{items.length > 0 && <div className="chat-recommendations"><span className="eyebrow">MẤY GỢI Ý ĐẦU TIÊN</span>{items.map((item) => <article className="chat-rec-card" key={item.id}><button onClick={() => openItem(item)} className="chat-rec-art" aria-label={`Xem ${item.title}`}><ChatPoster item={item}/></button><div className="chat-rec-info"><span className="eyebrow">{item.eyebrow}</span><button className="chat-rec-title" onClick={() => openItem(item)}><b>{item.title}</b></button><small>{item.year || item.duration} · {item.match}% hợp gu</small><p>{item.why}</p></div><button className={`icon-button ${saved.some((entry) => entry.id === item.id) ? 'on' : ''}`} onClick={() => toggleSaved(item)} title="Lưu vào danh sách"><Bookmark size={16}/></button></article>)}</div>}</div></div>
}

const traceStageLabels: Record<string, string> = {
  queue: 'Đang chờ', start: 'Tiếp nhận', embed: 'Embedding câu hỏi', session: 'Bộ nhớ hội thoại',
  intent: 'Phân tích ý định', resolve: 'Chuẩn hoá tham chiếu', retrieval: 'Truy xuất ứng viên',
  rerank: 'Xếp hạng lại', evidence: 'Tạo bằng chứng', answer: 'Sinh câu trả lời',
  answer_line: 'Dòng trả lời', persist: 'Lưu kết quả', done: 'Kết thúc',
}

function TraceDetails({ run }: { run: TraceRun }) {
  const statusLabels: Record<string, string> = {
    queued: 'Đang chờ', running: 'Đang xử lý', complete: 'Hoàn tất', partial: 'Hoàn tất một phần',
    error: 'Lỗi', interrupted: 'Bị gián đoạn',
  }
  const events = run.events || []
  const stages = new Map<string, TraceRun['events'][number]>()
  for (const event of events) if (event.stage !== 'answer_line') stages.set(event.stage, event)
  return <div className="trace-detail-content">
    <div className="trace-question"><span className="eyebrow">CÂU HỎI</span><p>{run.question}</p></div>
    <div className="trace-status-row"><span className={`trace-status status-${run.status}`}>{statusLabels[run.status] || run.status}</span><span>{String(run.trace_id)}</span></div>
    <div className="trace-stage-list">
      {[...stages.entries()].map(([stage, event]) => <div className="trace-stage" key={`${event.seq}-${stage}`}><span className={`trace-stage-mark stage-${event.status}`}/><div><b>{traceStageLabels[stage] || stage}</b><small>{event.status}{typeof event.duration_ms === 'number' ? ` · ${(event.duration_ms / 1000).toFixed(1)} giây` : ''}</small></div></div>)}
      {stages.size === 0 && <p className="trace-muted">Đang chờ worker bắt đầu xử lý…</p>}
    </div>
    {run.answer && <div className="trace-answer"><span className="eyebrow">CÂU TRẢ LỜI ĐANG TẠO</span><p>{run.answer}</p></div>}
    {run.error && <InlineError message={run.error}/>}
    {events.length > 0 && <details className="trace-events"><summary>Sự kiện truy vấn ({events.length})</summary><ol>{events.map((event) => <li key={event.seq}><b>{traceStageLabels[event.stage] || event.stage}</b><span>{event.status}</span>{event.text && <p>{event.text}</p>}</li>)}</ol></details>}
  </div>
}

function ChatPoster({ item }: { item: MediaItem }) {
  if (!item.image) return null
  return <img
    src={item.image}
    alt=""
    loading="lazy"
    decoding="async"
    onError={(event) => {
      const image = event.currentTarget
      if (item.fallbackImage && image.dataset.fallback !== 'true') {
        image.dataset.fallback = 'true'
        image.src = item.fallbackImage
      } else {
        image.remove()
      }
    }}
  />
}

function AuthPage({ mode, onSignIn }: { mode: 'login' | 'register'; onSignIn: (token: string) => void }) {
  const [email, setEmail] = useState('')
  const [password, setPassword] = useState('')
  const [confirm, setConfirm] = useState('')
  const [code, setCode] = useState('')
  const [recovery, setRecovery] = useState(false)
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState('')
  const [stage, setStage] = useState<'credentials' | 'verify' | 'enroll' | 'codes'>('credentials')
  const [preauth, setPreauth] = useState('')
  const [enrollment, setEnrollment] = useState<{ enrollment_token: string; totp_secret: string; qr_code_png_base64: string } | null>(null)
  const [codes, setCodes] = useState<string[]>([])
  const [showSecret, setShowSecret] = useState(false)
  const navigate = useNavigate()
  const location = useLocation()
  const afterAuth = (nextToken?: string) => {
    if (nextToken) { onSignIn(nextToken); const destination = (location.state as { from?: string } | null)?.from || '/chatbot'; navigate(destination, { replace: true }) }
    else navigate('/login', { replace: true })
  }
  const submitCredentials = async (event: FormEvent) => {
    event.preventDefault(); setError(''); setBusy(true)
    try {
      if (mode === 'login') { const result = await api.login(email, password); setPreauth(result.preauth_token); setStage('verify') }
      else {
        if (password !== confirm) throw new ApiError('Passwords do not match.', 422)
        const result = await api.register(email, password, confirm); setEnrollment(result); setStage('enroll')
      }
    } catch (reason) { setError((reason as Error).message) } finally { setBusy(false) }
  }
  const submitCode = async (event: FormEvent) => {
    event.preventDefault(); setError(''); setBusy(true)
    try {
      if (mode === 'login') {
        const result = await api.login2fa(preauth, code)
        afterAuth(result.access_token)
      } else if (enrollment) {
        const result = await api.confirmRegistration(enrollment.enrollment_token, code)
        setCodes(result.recovery_codes); setStage('codes')
      }
    } catch (reason) { setError((reason as Error).message) } finally { setBusy(false) }
  }
  const useRecoveryCode = () => { setRecovery((value) => !value); setCode('') }
  const copyCodes = async () => { await navigator.clipboard?.writeText(codes.join('\n')); setShowSecret(true) }
  const downloadCodes = () => {
    const file = new Blob([codes.join('\n')], { type: 'text/plain;charset=utf-8' })
    const url = URL.createObjectURL(file)
    const anchor = document.createElement('a')
    anchor.href = url
    anchor.download = 'ma-khoi-phuc-khoang-ranh.txt'
    anchor.click()
    URL.revokeObjectURL(url)
  }
  return <div className="auth-page"><div className="auth-art"><div className="auth-art-orbit"/><Link to="/" className="brand auth-brand"><span className="brand-name">GMRe</span><small className="brand-tagline">KHÁM PHÁ GIẢI TRÍ HỢP GU</small></Link><div className="auth-art-content"><span className="eyebrow eyebrow-light">TRẢI NGHIỆM GMRe CỦA BẠN</span><h1>Mỗi tâm trạng<br/>đều có một điều<br/><em>hợp để tận hưởng.</em></h1><p>Trò chuyện một chút, khám phá thêm nhiều điều bạn sẽ thích.</p><div className="auth-art-pills"><span>🎬 Một bộ phim hay</span><span>🎧 Giai điệu đúng mood</span><span>🎮 Cuộc vui cùng bạn</span></div></div><div className="auth-art-foot"><span>✳</span> Tìm điều vui, theo cách của bạn.</div></div>
    <div className="auth-form-side"><Link to="/" className="auth-back"><ArrowLeft size={15}/> Về trang chủ</Link><div className="auth-panel">
      {stage === 'credentials' && <><span className="eyebrow">{mode === 'login' ? 'CHÀO MỪNG BẠN QUAY LẠI' : 'BẮT ĐẦU MỘT TRẢI NGHIỆM GMRe MỚI'}</span><h2>{mode === 'login' ? 'Mừng bạn trở lại.' : 'Tạo góc riêng của bạn.'}</h2><p>{mode === 'login' ? 'Đăng nhập để tiếp tục những gợi ý hợp gu.' : 'Đăng ký nhanh để lưu lại những điều mình thích.'}</p><form onSubmit={submitCredentials} className="auth-fields"><label className="form-label">Địa chỉ email<input className="text-input" type="email" autoComplete="email" value={email} onChange={(event) => setEmail(event.target.value)} placeholder="ten@email.com" required/></label><label className="form-label">Mật khẩu<input className="text-input" type="password" autoComplete={mode === 'login' ? 'current-password' : 'new-password'} minLength={10} value={password} onChange={(event) => setPassword(event.target.value)} placeholder="Ít nhất 10 ký tự" required/></label>{mode === 'register' && <label className="form-label">Nhập lại mật khẩu<input className="text-input" type="password" autoComplete="new-password" minLength={10} value={confirm} onChange={(event) => setConfirm(event.target.value)} placeholder="Nhập lại mật khẩu" required/></label>}{error && <InlineError message={error}/>}<button className="button button-primary auth-submit" disabled={busy}>{busy ? 'Đang xử lý…' : mode === 'login' ? 'Tiếp tục đăng nhập' : 'Tạo tài khoản'} <ArrowRight size={16}/></button></form>{DEMO_MODE && <button type="button" className="demo-login-button" onClick={() => afterAuth(DEMO_TOKEN)}><Sparkles size={15}/> Dùng thử không cần tài khoản</button>}<div className="auth-switch">{mode === 'login' ? <>Chưa có tài khoản? <Link to="/register">Đăng ký miễn phí</Link></> : <>Đã có tài khoản? <Link to="/login">Đăng nhập</Link></>}</div><div className="auth-secure"><ShieldCheck size={15}/> Tài khoản được bảo vệ bằng xác thực hai bước</div></>}
      {stage === 'verify' && <><span className="eyebrow">THÊM MỘT LỚP BẢO VỆ</span><h2>Xác nhận là bạn nhé.</h2><p>Nhập mã {recovery ? 'khôi phục' : '6 chữ số'} từ ứng dụng xác thực của bạn.</p><form onSubmit={submitCode} className="auth-fields"><label className="form-label">{recovery ? 'Mã khôi phục' : 'Mã xác thực'}<input className={`text-input code-input ${!recovery ? 'numeric-code' : ''}`} inputMode={recovery ? 'text' : 'numeric'} autoComplete="one-time-code" maxLength={recovery ? 32 : 6} value={code} onChange={(event) => setCode(event.target.value)} placeholder={recovery ? 'XXXX-XXXX-XXXX-XXXX' : '000000'} required/></label>{error && <InlineError message={error}/>}<button className="button button-primary auth-submit" disabled={busy}>{busy ? 'Đang xác nhận…' : 'Xác nhận và đăng nhập'} <ArrowRight size={16}/></button></form><button className="text-action" onClick={useRecoveryCode}>{recovery ? 'Dùng mã xác thực 6 chữ số' : 'Dùng mã khôi phục'}</button><button className="text-action muted-action" onClick={() => { setStage('credentials'); setCode('') }}>Quay lại đăng nhập</button></>}
      {stage === 'enroll' && enrollment && <><span className="eyebrow">THIẾT LẬP TÀI KHOẢN AN TOÀN</span><h2>Quét mã để tiếp tục.</h2><p>Mở Google Authenticator hoặc ứng dụng tương tự, rồi quét mã QR bên dưới.</p><div className="qr-wrap"><img src={`data:image/png;base64,${enrollment.qr_code_png_base64}`} alt="Mã QR thiết lập xác thực hai bước"/></div><button className="text-action" onClick={() => setShowSecret((value) => !value)}>{showSecret ? 'Ẩn mã thiết lập thủ công' : 'Không quét được? Xem mã thiết lập'}</button>{showSecret && <code className="secret-code">{enrollment.totp_secret}</code>}<form onSubmit={submitCode} className="auth-fields"><label className="form-label">Nhập mã 6 chữ số để hoàn tất<input className="text-input code-input numeric-code" inputMode="numeric" maxLength={6} value={code} onChange={(event) => setCode(event.target.value)} required/></label>{error && <InlineError message={error}/>}<button className="button button-primary auth-submit" disabled={busy}>{busy ? 'Đang xác nhận…' : 'Hoàn tất đăng ký'} <Check size={16}/></button></form></>}
      {stage === 'codes' && <><span className="eyebrow">TÀI KHOẢN ĐÃ SẴN SÀNG</span><h2>Lưu lại mã khôi phục.</h2><p>Cất những mã này ở nơi an toàn. Mỗi mã chỉ dùng được một lần và không thể xem lại.</p><div className="recovery-codes">{codes.map((entry) => <code key={entry}>{entry}</code>)}</div><button className="button button-outline auth-submit" onClick={() => void copyCodes()}><Check size={16}/> Sao chép mã</button><button className="text-action" onClick={downloadCodes}>Tải mã khôi phục về thiết bị</button>{showSecret && <div className="success-note"><Check size={15}/> Đã sao chép mã khôi phục</div>}<button className="button button-primary auth-submit" onClick={() => afterAuth()}>Đến trang đăng nhập <ArrowRight size={16}/></button></>}
    </div><div className="auth-footnote">Tiếp tục nghĩa là bạn đồng ý với <span>Điều khoản sử dụng</span> và <span>Chính sách riêng tư</span>.</div></div></div>
}

function EmptyState({ icon, title, text, action }: { icon: ReactNode; title: string; text: string; action?: ReactNode }) { return <div className="empty-state"><span className="empty-icon">{icon}</span><h2>{title}</h2><p>{text}</p>{action}</div> }
function InlineError({ message }: { message: string }) { return <div className="inline-error" role="alert"><span>!</span>{message}</div> }

export default App
