import { memo } from 'react'
import { motion } from 'framer-motion'

const dots = [0, 1, 2]

export const TypingIndicator = memo(function TypingIndicator() {
  return (
    <motion.div
      initial={{ opacity: 0, y: 10 }}
      animate={{ opacity: 1, y: 0 }}
      exit={{ opacity: 0, y: -8 }}
      transition={{ duration: 0.18 }}
      className="flex justify-start"
    >
      <div className="rounded-3xl rounded-bl-lg border border-stone-200/80 bg-white px-4 py-3 shadow-sm">
        <div className="flex items-center gap-1.5">
          {dots.map((dot) => (
            <motion.span
              key={dot}
              className="h-2.5 w-2.5 rounded-full bg-amber-500/70"
              animate={{ y: [0, -4, 0], opacity: [0.45, 1, 0.45] }}
              transition={{
                duration: 0.8,
                repeat: Number.POSITIVE_INFINITY,
                delay: dot * 0.12,
                ease: 'easeInOut',
              }}
            />
          ))}
        </div>
      </div>
    </motion.div>
  )
})
