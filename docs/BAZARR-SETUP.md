# AI Subtitle Translator: Bazarr+ Setup Guide

Get AI-powered subtitle translation running alongside your existing Bazarr+ installation in under 2 minutes.

## Automatic install

The install script detects your Bazarr+ container, picks the right networking, and prints the encryption key:

```bash
curl -sSL https://raw.githubusercontent.com/LavX/ai-subtitle-translator/main/install.sh | bash
```

It auto-detects whether Bazarr+ uses host networking, a custom bridge, or default bridge, and configures accordingly. At the end it prints the service URL and encryption key to copy into Bazarr+.

## Manual setup

If you prefer to run it manually:

```bash
docker run -d \
  --name ai-subtitle-translator \
  --restart unless-stopped \
  -p 8765:8765 \
  -v ai-subtitle-translator-data:/app/data \
  ghcr.io/lavx/ai-subtitle-translator:latest
```

That's it. The service starts on port 8765, generates an encryption key, and creates a persistent database automatically.

If Bazarr+ and the translator are on the same Docker network (see [Networking options](#networking-options)), you can skip `-p 8765:8765` and use the container name instead.

## Get your encryption key

```bash
docker exec ai-subtitle-translator cat /app/data/encryption.key
```

Copy the 64-character hex string. You'll paste it into Bazarr+ next.

## Configure Bazarr+

1. Open Bazarr+ Settings
2. Go to the AI Subtitle Translator provider section
3. Set the following:
   - **Translator URL**: `http://ai-subtitle-translator:8765` (if same Docker network) or `http://<host-ip>:8765`
   - **OpenRouter API Key**: Your key from [openrouter.ai/keys](https://openrouter.ai/keys)
   - **Encryption Key**: Run `docker exec ai-subtitle-translator cat /app/data/encryption.key` and paste the 64-character hex string
   - **Model**: Pick a model (see recommendations below)
4. Click **Test** to verify everything works
5. Save

## Verify it works

From the command line:

```bash
# Check service health
curl http://localhost:8765/health

# Test with your OpenRouter key (replace sk-or-... with your key)
curl -X POST http://localhost:8765/api/v1/test \
  -H "Content-Type: application/json" \
  -d '{"apiKey": "sk-or-v1-your-key-here"}'
```

You should see `"status": "ok"` for the API key check.

## Recommended models

| Model | Cost/episode | Speed | Best for |
|-------|-------------|-------|----------|
| `google/gemini-2.5-flash-lite-preview-09-2025` | ~$0.008 | Fast | Best value |
| `meta-llama/llama-4-maverick` | ~$0.02 | Fastest | Speed |
| `anthropic/claude-haiku-4.5` | ~$0.05 | Medium | Best accuracy |
| `inception/mercury-2` | ~$0.10 | Fast | Good balance |

Set the default model via environment variable:

```bash
docker run -d \
  --name ai-subtitle-translator \
  --restart unless-stopped \
  -p 8765:8765 \
  -e OPENROUTER_DEFAULT_MODEL=google/gemini-2.5-flash-lite-preview-09-2025 \
  -v ai-subtitle-translator-data:/app/data \
  ghcr.io/lavx/ai-subtitle-translator:latest
```

## Docker Compose

If you prefer docker-compose, add this to your existing stack:

```yaml
services:
  ai-subtitle-translator:
    image: ghcr.io/lavx/ai-subtitle-translator:latest
    container_name: ai-subtitle-translator
    restart: unless-stopped
    network_mode: host
    environment:
      - OPENROUTER_DEFAULT_MODEL=google/gemini-2.5-flash-lite-preview-09-2025
    volumes:
      - ai-subtitle-translator-data:/app/data

volumes:
  ai-subtitle-translator-data:
```

## Networking options

**Same Docker Compose stack (most common):** Add the translator to your existing `docker-compose.yml` (see above). Bazarr+ connects to `http://ai-subtitle-translator:8765` using the container name.

**Separate containers, same machine:** Use `-p 8765:8765` (the default one-liner). Bazarr+ connects to `http://localhost:8765` or `http://<host-ip>:8765`.

**Same Docker network:** If Bazarr+ is on a custom bridge network, join it:

```bash
docker run -d \
  --name ai-subtitle-translator \
  --restart unless-stopped \
  --network <Bazarr-network-name> \
  -v ai-subtitle-translator-data:/app/data \
  ghcr.io/lavx/ai-subtitle-translator:latest
```

Bazarr+ connects to `http://ai-subtitle-translator:8765`.

**Host network:** If your setup uses `--network host`:

```bash
docker run -d \
  --name ai-subtitle-translator \
  --restart unless-stopped \
  --network host \
  -v ai-subtitle-translator-data:/app/data \
  ghcr.io/lavx/ai-subtitle-translator:latest
```

Bazarr+ connects to `http://localhost:8765`.

**Different machines:** Expose with `-p 8765:8765`. Bazarr+ connects to `http://<translator-ip>:8765`. Encryption is recommended in this case (enabled by default).

## Disable encryption

If you don't need encryption (same machine, trusted network):

```bash
docker run -d \
  --name ai-subtitle-translator \
  --restart unless-stopped \
  -p 8765:8765 \
  -e ENCRYPTION_ENABLED=false \
  -v ai-subtitle-translator-data:/app/data \
  ghcr.io/lavx/ai-subtitle-translator:latest
```

Leave the encryption key field empty in Bazarr+.

## Troubleshooting

**Check logs:**
```bash
docker logs -f ai-subtitle-translator
```

**Service not starting:**
```bash
docker logs ai-subtitle-translator 2>&1 | head -20
```

**Permission errors on data volume:**
```bash
# Find the appuser UID inside the container
docker exec ai-subtitle-translator id
# Fix host permissions (replace 1000 with the UID from above)
sudo chown -R 1000:1000 /var/lib/docker/volumes/ai-subtitle-translator-data/_data
```

**Reset encryption key:**
```bash
docker exec ai-subtitle-translator python -m subtitle_translator.cli regenerate-key
docker restart ai-subtitle-translator
# Then get the new key
docker exec ai-subtitle-translator cat /app/data/encryption.key
```

## Links

- [AI Subtitle Translator GitHub](https://github.com/LavX/ai-subtitle-translator)
- [Bazarr+](https://github.com/LavX/Bazarr)
- [OpenRouter](https://openrouter.ai/) (get your API key here)
- [Full API docs](http://localhost:8765/docs) (when service is running)
