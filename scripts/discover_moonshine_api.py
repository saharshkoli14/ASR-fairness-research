"""Introspect the moonshine-voice API so the backend can be written against reality.

    pip install moonshine-voice
    python scripts/discover_moonshine_api.py
"""

import inspect

import moonshine_voice as mv

print("version:", getattr(mv, "__version__", "?"))
print("\ntop-level names:")
print("  ", [n for n in dir(mv) if not n.startswith("_")])

for name in ("download_model", "Transcriber", "TranscriptEventListener", "ModelArch", "load_wav_file"):
    obj = getattr(mv, name, None)
    print(f"\n=== {name}: {'MISSING' if obj is None else type(obj).__name__}")
    if obj is None:
        continue
    if inspect.isclass(obj):
        members = [m for m in dir(obj) if not m.startswith("_")]
        print("   members:", members)
        for meth in ("__init__", "transcribe_without_streaming", "start", "stop", "add_audio"):
            f = getattr(obj, meth, None)
            if f and callable(f):
                try:
                    print(f"   {meth}{inspect.signature(f)}")
                except (ValueError, TypeError):
                    pass
    elif callable(obj):
        try:
            print("   signature:", inspect.signature(obj))
        except (ValueError, TypeError):
            pass
        print("   doc:", (inspect.getdoc(obj) or "")[:400])

# ModelArch enumerates available architectures — this tells us how to ask for MEDIUM streaming.
arch = getattr(mv, "ModelArch", None)
if arch is not None:
    print("\nModelArch values:", [m for m in dir(arch) if not m.startswith("_")])
