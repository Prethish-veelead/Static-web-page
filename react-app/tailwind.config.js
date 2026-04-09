/** @type {import('tailwindcss').Config} */
export default {
  content: ['./index.html', './src/**/*.{js,ts,jsx,tsx}'],
  theme: {
    extend: {
      colors: {
        chat: {
          ink: '#1f2937',
          muted: '#6b7280',
          panel: 'rgba(255, 252, 247, 0.84)',
          border: 'rgba(120, 113, 108, 0.12)',
          user: '#1f2937',
          assistant: '#ffffff',
          accent: '#d97706',
        },
      },
      boxShadow: {
        chat: '0 24px 80px rgba(120, 53, 15, 0.16)',
      },
    },
  },
  plugins: [],
}
