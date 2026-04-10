# AirBall

Computer vision + AI basketball shooting coach.

## Monorepo Structure

- `client/`: Next.js frontend
- `Server/`: FastAPI backend + pose/shot detection pipeline

## Prerequisites

- Python 3.10+
- Node.js 20+
- npm

## Backend Setup (`Server`)

1. Create and activate a virtual environment.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Start the API:

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

### Backend Environment Variables

- `AIRBALL_ALLOWED_ORIGINS` (optional): comma-separated CORS origins.
	- Default: `http://localhost:3000`
	- Example: `AIRBALL_ALLOWED_ORIGINS=http://localhost:3000,https://example.com`

## Frontend Setup (`client`)

1. Install dependencies:

```bash
npm install
```

2. Configure env file:

```bash
# client/.env.local
NEXT_PUBLIC_API_URL=http://localhost:8000
```

3. Start frontend:

```bash
npm run dev
```

## API Contract

### `GET /health`

- Purpose: service health probe
- Response `200`:

```json
{
	"status": "ok"
}
```

### `GET /video_feed`

- Purpose: MJPEG stream for live camera + pose overlay
- Response `200`:
	- `Content-Type: multipart/x-mixed-replace; boundary=frame`
	- Body: MJPEG frame boundary stream

### `GET /`

- Purpose: minimal debug page that renders the stream
- Response `200`:
	- `Content-Type: text/html`

## Shot JSON Output

- Detected shots are written to `Server/Shots/`.
- Each file is named `shot_<uuid>.json`.

## Notes

- Keep full body joints in frame (especially hips/knees/ankles) for best metrics quality.
- Ball-in-hand evidence is used to gate coaching feedback confidence.
