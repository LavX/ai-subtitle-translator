"""What the GUI and the API show about back ends. Positions and states only, never keys."""


async def _fill(runtime, connection, row) -> None:
    try:
        backend = runtime.registry.get(connection.backend_type)
    except Exception:  # noqa: BLE001 - an unavailable type is already listed above
        return
    if backend.uses_sign_in:
        # The folders the engine gives this type (under BACKENDS_DATA_DIR), where the
        # connection's sign-in folder is.
        state = await backend.sign_in_status(connection, runtime.engine.context_for(connection.backend_type))
        row["signedIn"] = state.signed_in if state else None
    provider = runtime.engine.provider_for(connection) if connection.enabled else None
    keys = getattr(provider, "keys", None)
    if keys is not None:
        row["keys"] = [
            {"index": h.index, "state": str(h.state), "until": h.until} for h in keys.health(row["model"])
        ]


async def backend_status(runtime) -> dict:
    backends = []
    for entry in runtime.registry.entries():
        backend = entry.backend
        # No sign-in command is advertised: the image ships no wrapper to run one.
        uses_sign_in = bool(backend and backend.uses_sign_in)
        backends.append(
            {
                "type": entry.type_id,
                "name": backend.display_name if backend else entry.type_id,
                "source": "built-in" if entry.builtin else f"{entry.distribution or 'unknown'} {entry.version or ''}".strip(),
                "available": backend is not None,
                "error": entry.error,
                "usesSignIn": uses_sign_in,
            }
        )
    connections = []
    for connection in runtime.connections:
        row = {
            "id": connection.id,
            "type": connection.backend_type,
            "enabled": connection.enabled,
            "model": str(connection.settings.get("model", "")),
            "signedIn": None,
            "keys": [],
        }
        try:
            await _fill(runtime, connection, row)
        except Exception:  # noqa: BLE001 - one broken connection must not hide the others
            row["signedIn"], row["keys"] = None, []
        connections.append(row)
    return {"backends": backends, "connections": connections}
