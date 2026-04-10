# AirBall Client

Next.js frontend for AirBall.

## Development

1. Install dependencies:

```bash
npm install
```

2. Create env file:

```bash
# .env.local
NEXT_PUBLIC_API_URL=http://localhost:8000
```

3. Start dev server:

```bash
npm run dev
```

4. Open `http://localhost:3000`.

## Scripts

- `npm run dev`: run Next.js in development mode
- `npm run build`: production build
- `npm run start`: run production server
- `npm run lint`: run ESLint

## Frontend API Dependency

- Expects backend `GET /health` for status checks
- Expects backend `GET /video_feed` for MJPEG stream preview
