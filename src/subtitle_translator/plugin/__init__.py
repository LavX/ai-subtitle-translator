"""The Bazarr+ translator-kind entry. The SDK binding needs bazarr_plugin_sdk, which the host provides on the worker path."""


def __getattr__(name: str):
    if name in ("BuiltinTranslatorPlugin", "ExtensionTranslatorPlugin"):
        from subtitle_translator.plugin import sdk_binding

        return getattr(sdk_binding, name)
    raise AttributeError(name)
