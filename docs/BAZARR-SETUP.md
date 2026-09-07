# AI Subtitle Translator: Bazarr+ Setup Guide

Run AI Subtitle Translator alongside Bazarr+, then connect it using the shared encryption key and an OpenRouter API key.

## Automatic install

The install script detects containers named `bazarr` or `bazarr-ui-test`, configures networking, and prints the encryption key:

```bash
curl -sSL https://raw.githubusercontent.com/LavX/ai-subtitle-translator/main/install.sh | bash
```

It auto-detects whether Bazarr+ uses host networking, a custom bridge, or default bridge, and configures accordingly. At the end it prints a service URL and encryption key. Choose a URL reachable from Bazarr+ using the [Networking options](#networking-options) below; a `localhost` hint is not usable from a separate bridge-network container.

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
   - **Model**: Enter an explicit model such as `google/gemini-3.1-flash-lite:floor` (see recommendations below)
   - **Provider Routing**: Fastest (the default) picks the highest-throughput OpenRouter provider; Cheapest picks the lowest price. The `:nitro` and `:floor` variants also unlock OpenRouter's priority and flex tiers. The routing decides which provider bills the request, so the price per episode moves with it
4. Click **Test** to verify everything works
5. Save

## Verify it works

For the protected API check, first derive `AUTH_TOKEN` using the [README authentication command](../README.md#authentication). Then run these checks inside the translator container, which also works when no host port is published:

```bash
# Check service health
docker exec ai-subtitle-translator curl http://localhost:8765/health

# Test with your OpenRouter key (replace sk-or-... with your key)
docker exec ai-subtitle-translator curl -X POST http://localhost:8765/api/v1/test \
  -H "Content-Type: application/json" \
  -H "X-Auth-Token: $AUTH_TOKEN" \
  -d '{"apiKey": "sk-or-v1-your-key-here"}'
```

You should see `"status": "ok"` for the API key check.

## Model and timeout settings

Start with `google/gemini-3.1-flash-lite:floor` for the service examples. See the [current model comparison](../README.md#subtitle-translation-leaderboard) and [episode/season estimates](../README.md#episode-movie-and-season-cost-estimates) for benchmark results, Luna compatibility notes, and all tested candidates. Costs depend on dialogue, reasoning, provider route and retries.

Set the default model via environment variable:

```bash
docker run -d \
  --name ai-subtitle-translator \
  --restart unless-stopped \
  -p 8765:8765 \
  -e OPENROUTER_DEFAULT_MODEL=google/gemini-3.1-flash-lite:floor \
  -v ai-subtitle-translator-data:/app/data \
  ghcr.io/lavx/ai-subtitle-translator:latest
```

The default HTTP timeout is 120 seconds. For slow `:floor` providers, including the longer DeepSeek benchmark runs, set `REQUEST_TIMEOUT=600` in the container environment. This is not a whole-job deadline.

## Docker Compose

If you prefer docker-compose, add this to your existing stack:

```yaml
services:
  ai-subtitle-translator:
    image: ghcr.io/lavx/ai-subtitle-translator:latest
    container_name: ai-subtitle-translator
    restart: unless-stopped
    ports:
      - "8765:8765"
    environment:
      - OPENROUTER_DEFAULT_MODEL=google/gemini-3.1-flash-lite:floor
    volumes:
      - ai-subtitle-translator-data:/app/data

volumes:
  ai-subtitle-translator-data:
```

## Networking options

**Shared Compose network:** Add the translator to your stack and ensure both services join the same Compose network. Bazarr+ connects to `http://ai-subtitle-translator:8765`. If Bazarr+ uses host networking or a separate network, use the appropriate host-IP option below.

**Separate containers, same machine:** Use `-p 8765:8765` (the default one-liner). For separate bridge networks, Bazarr+ connects to `http://<host-ip>:8765`. Container-local `localhost` points back to Bazarr+, not the translator.

**Same Docker network:** If Bazarr+ is on a custom bridge network, replace `YOUR_BAZARR_NETWORK` with its name and join it:

```bash
docker run -d \
  --name ai-subtitle-translator \
  --restart unless-stopped \
  --network YOUR_BAZARR_NETWORK \
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

Use `http://localhost:8765` only when Bazarr+ also uses host networking or runs directly on the same host. Otherwise use that host's reachable IP.

**Different machines:** Expose with `-p 8765:8765`. Bazarr+ connects to `http://<translator-ip>:8765`. Use HTTPS on an untrusted network. API-key encryption is enabled by default, but it does not encrypt subtitle text or the authentication token.

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

Leave the encryption key field empty in Bazarr+. This also disables the shared-token check on translation and job endpoints.

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
- [Bazarr+](https://github.com/LavX/bazarr)
- [OpenRouter](https://openrouter.ai/) (get your API key here)
- [Full API docs](http://localhost:8765/docs) (when service is running)
