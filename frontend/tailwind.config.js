/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        primary: '#0A0E27',
        secondary: '#6366F1',
        accent: '#8B5CF6',
      },
    },
  },
  plugins: [],
}
