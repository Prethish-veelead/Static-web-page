import { ChatWindow } from './components/chat/ChatWindow'

function App() {
  return (
    <main className="relative min-h-screen overflow-hidden bg-[radial-gradient(circle_at_top,_rgba(255,255,255,0.95),_rgba(243,236,225,0.92)_38%,_rgba(225,214,193,0.88)_100%)] px-4 py-6 text-stone-900 sm:px-6 sm:py-8">
      <div className="pointer-events-none absolute inset-0 overflow-hidden">
        <div className="absolute left-1/2 top-[-8rem] h-72 w-72 -translate-x-1/2 rounded-full bg-amber-200/35 blur-3xl" />
        <div className="absolute right-[-4rem] top-24 h-56 w-56 rounded-full bg-rose-200/30 blur-3xl" />
        <div className="absolute bottom-[-5rem] left-[-3rem] h-64 w-64 rounded-full bg-orange-100/55 blur-3xl" />
      </div>

      <div className="relative mx-auto flex min-h-[calc(100vh-3rem)] w-full max-w-3xl items-stretch">
        <ChatWindow />
      </div>
    </main>
  )
}

export default App
