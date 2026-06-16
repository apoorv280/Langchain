from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from typing import TypedDict, Annotated, List
# from ChatModels.chatmodel_huggingface_api import LLM

load_dotenv()
LLM = HuggingFaceEndpoint(
    repo_id="meta-llama/Meta-Llama-3-8B-Instruct",
    task="text-generation"
)

model1  = ChatHuggingFace(llm = LLM)

#schema

class Review(TypedDict):

    key_theme: Annotated[list[str], 'write key theme discussed in the review']
    summary: Annotated[str, 'A brief summary of the review']
    sentiment: Annotated[str,'Return sentiment of the review']
    pros: Annotated[list[str],'write the advantages inside a list']
    cons: Annotated[list[str],'write the disadvantages inside a list']
    name: Annotated[str, 'write the creator of the review']

structured_model = model1.with_structured_output(Review)

# result = structured_model.invoke("The ceiling fan is great but it makes a lot of noise when it is on high speed. The customer service is also very slow and takes a lot of time to respond. Overall, I am not satisfied with the product and the service.")

result = structured_model.invoke("""Since you’ve been spending a lot of time working across laptops and diving into technical deep learning projects recently, let’s do a highly practical, technical product review of a tool that fits right into that high-focus workflow: the **Sony WH-1000XM5 Noise-Canceling Headphones**.

Instead of a generic overview, let's break this down into engineering fundamentals, real-world utility, and the annoying quirks that actually matter after long-term daily use.

---

## 🎧 The Sony WH-1000XM5: Long-Term Deep Dive

The XM5s represent a massive mechanical departure from the previous kings of the mountain (the XM4s). Sony redesigned the physical chassis, swapped the internal driver architecture, and doubled down on algorithmic processing.

### The Spec Sheet Blueprint

* **Weight:** 246.1 grams (noticeably lighter than Apple AirPods Max at 387.8g).
* **Driver Size:** 30mm carbon-fiber composite driver (down-sized from the XM4's 40mm, but stiffer).
* **Processors:** Dual-chip architecture (Sony V1 Integrated Processor + HD Noise Cancelling Processor QN1).
* **Microphones:** 8 beamforming mics total (4 per ear cup) for ANC and voice extraction.
* **Battery:** 30 hours (ANC On) / up to 50+ hours (ANC Off). USB-C fast charging gives 3 hours of playback on a 3-minute juice.

---

## What Makes Them Elite (The Pros)

### 1. High-Frequency Noise Isolation & ANC

Most Active Noise Cancellation (ANC) systems struggle with sudden, high-frequency spikes (like an office door slamming or keyboards clattering). The XM5 relies heavily on physical passive isolation combined with its dual-chip algorithm.

* **The Reality:** It attenuates low-frequency engine rumbles and cafe chatter by roughly **30dB**. If you work in a noisy environment or need to block out household distractions while debugging code, it drops ambient noise to about an eighth of its perceived volume.

### 2. Microphones and Call Clarity

If you are hopping onto Zoom or Teams calls frequently, this is where the XM5 completely outclasses its predecessors. The 4 beamforming mics utilize an AI-driven noise-reduction algorithm trained on over 500 million voice samples. It isolates your voice and cancels out severe wind or background clutter with incredible accuracy.

### 3. All-Day Comfort Structure

At ~246 grams, the pressure distribution is excellently tuned. Sony uses a "soft fit" vegan leatherette that provides enough friction to prevent sliding without pinching the sides of your head. The ear cups are deeper than previous iterations, preventing your ears from pressing flat against the inner driver grill during a 6-hour coding marathon.

---

## The Annoying Realities (The Cons)

### 1. The "No-Fold" Chassis Design (The Biggest Con)

Unlike the XM4s, **the XM5s do not fold down into a compact ball.** The arms use a stepless friction rod mechanism (similar to premium studio cans).

* **The Trade-off:** The travel case is massive—it takes up significant real estate in a tech backpack. If you are constantly tossing your headphones between a work bag and a personal bag, the bulk is irritating. Furthermore, the friction hinges have shown a higher vulnerability to stress-snapping over long-term aggressive use.

### 2. Mediocre Out-of-the-Box Sound Tuning

If you are an audiophile looking for flat, analytical reference sound, you won't find it here out of the box. The default factory tuning is notoriously "muddy"—the lower mids are bloated, and the treble lacks crisp definition.

* *The Fix:* You **must** download the Sony Sound Connect App and manually adjust the 5-band Equalizer. Dialing down the "Clear Bass" slightly and boosting the 2.5kHz and 16kHz bands transforms these into a highly dynamic, incredibly fun sounding pair of headphones.

### 3. Moisture Sensitivity & "Ghost Touches"

The capacitive touch panel on the right ear cup controls volume and track skipping. However, if you wear these in heavy rain or try to use them as gym buddies, sweat and condensation build-up inside the ear cup can confuse the proximity sensor, causing your music to randomly pause or skip.

---

## The Verdict: 8.5 / 10

| Best Suited For | Avoid If |
| --- | --- |
| Focus-heavy deep work, office commuting, endless Zoom calls, and people who hate ambient distraction. | Gym-goers who sweat heavily, minimalists with tiny commuter bags, or pure audio purists who refuse to use software EQs. |

The Sony WH-1000XM5s are a masterclass in software-driven audio engineering. They aren't perfect mechanical hardware pieces, but if your primary goal is putting your head down, blocking out the world, and getting work done across your laptops seamlessly, they are still incredibly tough to beat.
                                  review by Apoorv Pandey""")

# print(result)
print(result.keys())
# print(result1['key_theme'])
# print(result1['summary'])
# print(result1['sentiment'])
# print(result1['pros'])
# print(result1['cons'])
print(result['name'])


