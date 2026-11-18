"""The Bazarr+ translator-kind entry. Needs the [plugin] extra for the SDK binding."""


def __getattr__(name: str):
    if name in ("BuiltinTranslatorPlugin", "ExtensionTranslatorPlugin"):
        from subtitle_translator.plugin import sdk_binding

        return getattr(sdk_binding, name)
    raise AttributeError(name)
