export type User = { id: string; email: string }
export type MediaKind = 'movie' | 'game' | 'music' | 'book' | 'podcast' | 'activity'
export type MediaItem = {
  id: string
  title: string
  kind: MediaKind
  eyebrow: string
  mood: string
  duration: string
  match: number
  description: string
  shortDescription?: string
  why: string
  emoji: string
  image: string
  fallbackImage?: string
  tags: string[]
  year?: string
  platform?: string
}
export type Conversation = {
  id: string
  title: string
  created_at: string
  updated_at: string
  messages?: Message[]
}
export type Message = {
  id: string
  role: 'user' | 'assistant'
  content: string
  meta?: { items?: ApiRecommendation[]; [key: string]: unknown } | null
  created_at: string
}
export type TraceEvent = {
  seq: number
  stage: string
  status: string
  at?: string
  elapsed_ms?: number
  duration_ms?: number
  text?: string
  error?: string
  [key: string]: unknown
}
export type TraceRun = {
  trace_id: string
  conversation_id: string
  question: string
  status: string
  answer?: string
  active_stage?: string
  error?: string
  events: TraceEvent[]
  [key: string]: unknown
}
export type ApiRecommendation = {
  id?: string | number
  item_id?: string | number
  title?: string
  domain?: string
  score?: number
  evidence?: string
  shared_themes?: string[]
  shared_genres?: string[]
  [key: string]: unknown
}
