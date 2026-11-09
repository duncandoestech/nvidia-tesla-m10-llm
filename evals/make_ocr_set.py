#!/usr/bin/env python3
"""Generate a synthetic OCR set with known ground truth (sentences, numbers, code, receipts; some blur/rotation/noise)."""
import json, random
from PIL import Image, ImageDraw, ImageFilter, ImageFont
random.seed(7)
OUT = "evals/data/ocr"
SANS = "/usr/share/fonts/adwaita-sans-fonts/AdwaitaSans-Regular.ttf"
MONO = "/usr/share/fonts/liberation-mono-fonts/LiberationMono-Regular.ttf"
words = ("the quick brown fox jumps over lazy dog server memory cache tensor model token graph kernel driver power "
         "thermal fan voltage switch bridge network packet socket thread process signal buffer vector matrix").split()
def sentence(): return " ".join(random.choice(words) for _ in range(random.randint(6, 10))).capitalize() + "."
def numbers(): return f"Order #{random.randint(10000,99999)}  SKU {random.choice('ABCDEFGH')}{random.randint(100,999)}-{random.randint(10,99)}  Qty {random.randint(1,48)}  ${random.randint(1,999)}.{random.randint(0,99):02d}"
def code(): return random.choice(["def add(a, b):\n    return a + b", "for i in range(10):\n    print(i * 2)", "if x > 0:\n    y = math.sqrt(x)\nelse:\n    y = 0", "items = [n for n in data if n % 3 == 0]\ntotal = sum(items)"])
def receipt():
    lines = [f"{random.choice(['Coffee','Bagel','Milk','Eggs','Bread','Apples','Rice','Tea'])} x{random.randint(1,5)}  {random.randint(1,30)}.{random.randint(0,99):02d}" for _ in range(4)]
    return "\n".join(lines + [f"TOTAL  {random.randint(20,150)}.{random.randint(0,99):02d}"])
kinds = [("sentence", sentence, SANS), ("numbers", numbers, SANS), ("code", code, MONO), ("receipt", receipt, MONO)]
meta = []
for i in range(24):
    kind, fn, font = kinds[i % 4]
    text = fn() if kind != "sentence" else sentence() + "\n" + sentence()
    size = random.choice([22, 26, 30])
    f = ImageFont.truetype(font, size)
    w, h = 900, 60 + size * 2 * (text.count("\n") + 1)
    img = Image.new("RGB", (w, h), "white"); d = ImageDraw.Draw(img)
    d.multiline_text((30, 30), text, fill="black", font=f, spacing=size // 2)
    effect = ["clean", "blur", "rotate", "noise"][i // 4 % 4]
    if effect == "blur": img = img.filter(ImageFilter.GaussianBlur(1.2))
    if effect == "rotate": img = img.rotate(random.choice([-3, -2, 2, 3]), expand=True, fillcolor="white")
    if effect == "noise":
        px = img.load()
        for _ in range(w * h // 40): px[random.randrange(img.width), random.randrange(img.height)] = (random.randint(0, 120),) * 3
    name = f"ocr_{i:02d}_{kind}_{effect}.png"; img.save(f"{OUT}/{name}")
    meta.append({"file": name, "kind": kind, "effect": effect, "text": text})
json.dump(meta, open(f"{OUT}/truth.json", "w"), indent=1)
print(len(meta), "images")
