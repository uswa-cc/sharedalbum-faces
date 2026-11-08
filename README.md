# SharedAlbum face server (Render)

Finds faces in photos and turns each face into 512 numbers for the SharedAlbum
website, using InsightFace's lighter `buffalo_s` model so it fits Render's
free plan.

- `GET /health`: checks that the server is running
- `POST /detect`: send a photo as `file`; returns each face's box, quality and embedding

Photos are processed in memory and not saved.
