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
  why: string
  emoji: string
  image: string
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
