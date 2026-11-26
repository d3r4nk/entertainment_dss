import type { MediaItem } from './types'

const mediaApiBase = (import.meta.env.VITE_API_BASE_URL || '/api').replace(/\/+$/, '')
const steamArtwork = (appId: number) => `https://cdn.cloudflare.steamstatic.com/steam/apps/${appId}/library_600x900.jpg`

export const categories = [
  { id: 'all', label: 'Tất cả', icon: '✦', available: true },
  { id: 'movie', label: 'Phim & Series', icon: '▣', available: true },
  { id: 'game', label: 'Game', icon: '⌘', available: true },
  { id: 'activity', label: 'Hoạt động', icon: '↗', available: false },
  { id: 'podcast', label: 'Podcast', icon: '♫', available: false },
  { id: 'music', label: 'Âm nhạc', icon: '♪', available: false },
  { id: 'book', label: 'Sách', icon: '▤', available: false },
] as const

export const vibes = ['Thư giãn', 'Tập trung', 'Muốn cười', 'Phiêu lưu', 'Ở bên nhau']

export const catalogue: MediaItem[] = [
  {
    id: 'past-lives', title: 'Past Lives', kind: 'movie', eyebrow: 'PHIM · TÌNH CẢM', mood: 'Lắng đọng',
    duration: '1 giờ 46 phút', match: 96, year: '2023', platform: 'Netflix', emoji: '🌅',
    image: 'https://image.tmdb.org/t/p/w500/mJ0IJDj6T2YEgbkDHCfZE0fAsWA.jpg',
    description: 'Hai người bạn thời thơ ấu gặp lại nhau sau nhiều năm xa cách, trong một câu chuyện dịu dàng về những ngã rẽ của cuộc đời.',
    why: 'Một lựa chọn tinh tế, giàu cảm xúc cho buổi tối yên tĩnh và những câu chuyện còn đọng lại sau khi phim kết thúc.', tags: ['Tình cảm', 'Chậm rãi', 'Nhiều suy ngẫm'],
  },
  {
    id: 'journey', title: 'Journey', kind: 'game', eyebrow: 'GAME · PHIÊU LƯU', mood: 'Muốn thả lỏng',
    duration: '2–3 giờ', match: 94, year: '2012', platform: 'PlayStation · PC', emoji: '🏜️',
    image: steamArtwork(638230),
    description: 'Một hành trình ngắn qua sa mạc rộng lớn, nơi âm nhạc và khung cảnh kể chuyện thay cho lời thoại.',
    why: 'Không áp lực thắng thua, vừa đủ kỳ diệu để bạn nghỉ ngơi và khám phá theo nhịp riêng.', tags: ['Khám phá', 'Nhẹ nhàng', 'Một người chơi'],
  },
  {
    id: 'perfect-days', title: 'Perfect Days', kind: 'movie', eyebrow: 'PHIM · ĐỜI THƯỜNG', mood: 'Thư giãn',
    duration: '2 giờ 4 phút', match: 91, year: '2023', platform: 'MUBI', emoji: '🌿',
    image: 'https://image.tmdb.org/t/p/w500/tvUHVSTJV9ITON3oyHaWp7oaAc8.jpg',
    description: 'Một người đàn ông tìm thấy vẻ đẹp trong những nghi thức giản dị của đời sống thường ngày ở Tokyo.',
    why: 'Nhịp phim bình thản giúp bạn chậm lại, để ý đến những điều nhỏ xíu mà dễ thương.', tags: ['Ấm áp', 'Tối giản', 'Đời thường'],
  },
  {
    id: 'stardew', title: 'Stardew Valley', kind: 'game', eyebrow: 'GAME · MÔ PHỎNG', mood: 'Thư giãn',
    duration: 'Chơi theo nhịp riêng', match: 95, year: '2016', platform: 'PC · Console · Mobile', emoji: '🌱',
    image: '/image/stardew-valley.jpg',
    description: 'Bắt đầu cuộc sống mới ở nông trại nhỏ: trồng trọt, kết bạn và khám phá thung lũng.',
    why: 'Ấm áp, dễ chơi từng chút một và luôn có việc vui để làm cùng bạn bè.', tags: ['Co-op', 'Sáng tạo', 'Không áp lực'],
  },
  {
    id: 'overcooked', title: 'Overcooked! 2', kind: 'game', eyebrow: 'GAME · CO-OP', mood: 'Muốn cười',
    duration: '20 phút mỗi màn', match: 93, year: '2018', platform: 'PC · Console', emoji: '🍳',
    image: steamArtwork(728880),
    description: 'Cùng nhau chạy bếp trong những màn chơi hỗn loạn, nơi phối hợp tốt quan trọng hơn nấu đẹp.',
    why: 'Lý tưởng cho nhóm bạn muốn cười nhiều, nói nhiều và không ngại làm cháy vài chiếc bánh.', tags: ['Chơi cùng bạn', 'Hài hước', 'Nhịp nhanh'],
  },
  {
    id: 'blue-giant', title: 'Blue Giant', kind: 'movie', eyebrow: 'PHIM · ÂM NHẠC', mood: 'Muốn được tiếp sức',
    duration: '2 giờ', match: 90, year: '2023', platform: 'MUBI', emoji: '🎷',
    image: 'https://image.tmdb.org/t/p/w500/7LxbzvDvdeU7lBzB92JShURa0cM.jpg',
    description: 'Một chàng trai trẻ theo đuổi giấc mơ jazz với tất cả sức trẻ và sự quyết tâm.',
    why: 'Câu chuyện giàu năng lượng về đam mê, tình bạn và hành trình theo đuổi ước mơ.', tags: ['Âm nhạc', 'Nhiệt huyết', 'Tình bạn'],
  },
]

export const starterPrompts = [
  { icon: '🎬', title: 'Phim tâm lý hay', text: 'Gợi ý phim tâm lý hay, dưới 2 tiếng để xem một mình tối nay' },
  { icon: '🏜️', title: 'Game thư giãn', text: 'Mình muốn một game khám phá nhẹ nhàng, không áp lực thắng thua' },
  { icon: '🎮', title: 'Game co-op cuối tuần', text: 'Tụi mình có 3 người, gợi ý game co-op cuối tuần vui mà không quá khó' },
  { icon: '🌿', title: 'Phim nhẹ nhàng', text: 'Gợi ý một bộ phim ấm áp để xem cùng bạn, khoảng 2 tiếng' },
]

export const itemById = (id: string) => catalogue.find((item) => item.id === id)

function recommendationKind(value: unknown): 'movie' | 'game' | undefined {
  const domain = String(value || '').trim().toLowerCase()
  if (['game', 'games', 'video game', 'video_game'].includes(domain)) return 'game'
  if (['movie', 'movies', 'film', 'films'].includes(domain)) return 'movie'
  return undefined
}

export function recommendationToMedia(item: Record<string, unknown>, index: number): MediaItem {
  const reportedKind = [item.domain, item.kind, item.media_type, item.type]
    .map(recommendationKind)
    .find((kind) => kind !== undefined)
  const title = String(item.title || 'Gợi ý dành cho bạn')
  const knownItem = catalogue.find((entry) =>
    entry.title.toLocaleLowerCase('en') === title.trim().toLocaleLowerCase('en')
    && (!reportedKind || entry.kind === reportedKind),
  )
  const kind = reportedKind || knownItem?.kind || 'movie'
  const rawScore = Number(item.score)
  const match = Number.isFinite(rawScore) ? Math.max(70, Math.min(99, Math.round(rawScore <= 1 ? rawScore * 100 : rawScore))) : 91 - index
  const themes = Array.isArray(item.shared_themes) ? item.shared_themes.map(String) : []
  const genres = Array.isArray(item.shared_genres) ? item.shared_genres.map(String) : []
  const rawId = String(item.id ?? item.item_id ?? '')
  const hasNumericId = /^[1-9]\d{0,9}$/.test(rawId)
  return {
    id: `${kind}-${item.id ?? item.item_id ?? title}`,
    title,
    kind,
    eyebrow: knownItem?.eyebrow || (reportedKind ? `${kind === 'game' ? 'GAME' : 'PHIM'} · GỢI Ý` : 'GỢI Ý'),
    mood: typeof item.mood === 'string' ? item.mood : knownItem?.mood || '',
    duration: typeof item.duration === 'string' ? item.duration : knownItem?.duration || (kind === 'game' ? 'Thời lượng linh hoạt' : 'Chưa rõ thời lượng'),
    match,
    year: typeof item.year === 'string' || typeof item.year === 'number' ? String(item.year) : knownItem?.year,
    platform: typeof item.platform === 'string' ? item.platform : knownItem?.platform,
    emoji: knownItem?.emoji || (kind === 'game' ? '🎮' : '🎬'),
    tags: [...themes.slice(0, 2), ...genres.slice(0, 2)].length ? [...themes.slice(0, 2), ...genres.slice(0, 2)] : knownItem?.tags || [],
    why: String(item.evidence || 'Dựa trên chủ đề bạn đang quan tâm và những lựa chọn hợp gu trong danh mục.'),
    description: String(item.overview || item.description || knownItem?.description || ''),
    image: reportedKind && hasNumericId ? `${mediaApiBase}/media/${reportedKind}/${rawId}` : knownItem?.image || '',
    fallbackImage: knownItem?.image,
  }
}
