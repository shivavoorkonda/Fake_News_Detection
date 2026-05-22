"""
augmentation.py — Style-Adversarial Data Augmentation (SADA)

This module provides functions to procedurally generate style-adversarial articles:
1. Sober Fakes: Articles about classic fake/conspiracy/pseudoscience topics written
   in an extremely dry, objective, Reuters-style reporting tone.
2. Sensational Reals: True news articles augmented with sensationalized, emotional,
   and blog-style clickbait prefixes/suffixes.

By injecting these samples into the training dataset, we break the stylistic correlation
(formal = real, conversational/sensational = fake) and force DistilBERT to learn semantic context.
"""

import random
import logging
import pandas as pd

logger = logging.getLogger(__name__)

# Standard topics for Sober Fake News
TOPICS = [
    {
        "short": "crystal pyramid in Bermuda Triangle",
        "noun": "a massive crystal pyramid resting at the bottom of the ocean floor in the Bermuda Triangle",
        "details": "constructed from an unknown mineral that emits high-frequency energy pulses, which researchers believe are responsible for regional electromagnetic interference",
        "domain": "oceanography and marine physics"
    },
    {
        "short": "unidentified submerged objects in Mariana Trench",
        "noun": "multiple unidentified submerged objects demonstrating advanced acoustic properties inside the Mariana Trench",
        "details": "operating at depths exceeding 11,000 meters and moving at velocities inconsistent with modern thermal propulsion systems",
        "domain": "deep-sea acoustics and maritime defense"
    },
    {
        "short": "cold fusion in private laboratory",
        "noun": "a functional room-temperature cold fusion reactor constructed in a privately funded facility",
        "details": "producing a consistent net-positive energy yield of 120 percent using standard heavy water and palladium-coated nickel lattices",
        "domain": "nuclear engineering and condensed matter physics"
    },
    {
        "short": "weather modification base in Alaska",
        "noun": "a secret atmospheric research and ionospheric weather modification installation in eastern Alaska",
        "details": "transmitting high-frequency radio waves to alter local jet stream patterns, reportedly causing highly anomalous precipitation events",
        "domain": "meteorological defense and atmospheric research"
    },
    {
        "short": "metallic computer mechanism in Antarctic ice",
        "noun": "an ancient metallic mechanical computer preserved in deep Antarctic ice sheets",
        "details": "dating back approximately 12,000 years and showing sophisticated geared mathematical calibration for astronomical prediction",
        "domain": "archaeology and mechanical paleography"
    },
    {
        "short": "gravity distortion drive in Nevada",
        "noun": "a prototype gravity distortion propulsion system undergoing low-altitude testing in restricted airspace",
        "details": "manipulating local space-time metrics to achieve high-velocity acceleration without generating detectable thermal or kinetic drag",
        "domain": "aerospace propulsion and general relativity"
    },
    {
        "short": "lost subterranean city under the Andes",
        "noun": "a massive subterranean metropolitan complex stretching beneath the southern Andes range",
        "details": "containing structurally advanced stone chambers and complex water distribution channels built using unknown geological cutting techniques",
        "domain": "subterranean geology and ancient architecture"
    },
    {
        "short": "neural mind-reading device",
        "noun": "a direct electromagnetic neural interface system capable of reconstructing verbal thoughts at a distance",
        "details": "utilizing high-sensitivity quantum magnetometers to translate synaptic electrical activity into structured linguistic streams",
        "domain": "cognitive neurology and computational linguistics"
    },
    {
        "short": "miracle herbal compound for viruses",
        "noun": "an organic plant extract that completely neutralizes all active viral pathogens in under 48 hours",
        "details": "isolated from a rare high-altitude shrub and demonstrated in double-blind trials to inhibit cellular viral entry without toxicity",
        "domain": "pharmacognosy and molecular virology"
    },
    {
        "short": "alien biological entities in high-altitude crash",
        "noun": "multiple biological specimens of anomalous, non-human origin recovered from a high-altitude balloon incident",
        "details": "possessing a tri-helical genetic structure and unique metabolic pathways that utilize heavy metals rather than organic carbon",
        "domain": "exobiology and cellular genetics"
    }
]

# Formal reporting headline templates
SOBER_HEADLINE_TEMPLATES = [
    "Declassified documents confirm discovery of {noun}.",
    "Official reports indicate {noun} has been verified.",
    "Leaked scientific study details recovery of {noun}.",
    "Federal agencies reportedly investigating {noun} under strict security.",
    "According to regional authorities, {noun} was recently recorded.",
    "Technical assessment validates parameters of {noun}."
]

# Formal, objective body templates (Reuters wire style)
SOBER_BODY_TEMPLATES = [
    "A top-secret government oceanography and defense team has reportedly discovered {noun}. According to leaked documents obtained by independent journalists on Tuesday, the findings suggest that the system is {details}. While senior officials have repeatedly declined to comment publicly, independent analysts state that the scientific data could fundamentally challenge modern understanding of {domain}. The documents indicate that further investigations are currently underway under strict security protocols.",
    "In a detailed reports and assessments released by independent researchers, evidence has reportedly emerged confirming the existence of {noun}. The document, which has circulated among regional academic committees, suggests that the discovery is {details}. Regional administrators have issued a brief statement urging caution, though they did not deny the validity of the leaked papers. Senior researchers in {domain} are calling for an immediate international inquiry.",
    "Leaked diplomatic cables obtained by journalists reportedly describe the successful deployment of {noun}. According to sources close to the department, the project has been operating in a restricted sector for several years. The cables suggest the system is {details}. Neither the department nor local security agencies have responded to official requests for comment. Experts in {domain} remain divided on the long-term strategic implications.",
    "According to official files compiled by regional intelligence boards, researchers have finalized an investigation into {noun}. The analysis indicates the subject is {details}. Representatives stated that while the technology is still in experimental phases, it shows consistent performance. Further academic peer reviews in {domain} are scheduled for late next month."
]

def generate_sober_fakes(num_samples: int = 400, seed: int = 42) -> pd.DataFrame:
    """Generate a DataFrame of sober-toned fake news articles.

    Args:
        num_samples: Number of articles to generate.
        seed: Random seed for reproducibility.

    Returns:
        pd.DataFrame with columns: title, text, label (0 = FAKE), subject ('politics')
    """
    random.seed(seed)
    data = []

    for i in range(num_samples):
        topic = random.choice(TOPICS)
        headline_tmpl = random.choice(SOBER_HEADLINE_TEMPLATES)
        body_tmpl = random.choice(SOBER_BODY_TEMPLATES)

        # Build headline and body
        title = headline_tmpl.format(noun=topic["noun"])
        text = body_tmpl.format(
            noun=topic["noun"],
            details=topic["details"],
            domain=topic["domain"]
        )

        # Capitalize the first letter of the headline
        title = title[0].upper() + title[1:]

        data.append({
            "title": title,
            "text": text,
            "label": 0,  # FAKE
            "subject": "politics"
        })

    logger.info("Generated %d Sober Fake articles.", len(data))
    return pd.DataFrame(data)

# Sensational clickbait style components
CLICKBAIT_HEADLINE_PREFIXES = [
    "OMG! ", "BREAKING SHOCKER: ", "YOU WON'T BELIEVE THIS! ",
    "TRUTH EXPOSED: ", "This is absolutely embarrassing for them! ",
    "MASSIVE SCANDAL! ", "ALERT: ", "HILARIOUS! ", "CRITICAL UPDATE: "
]

CLICKBAIT_BODY_PREFIXES = [
    "Everyone is completely losing their minds over this! ",
    "This is a massive scandal that they tried to cover up! You won't believe what happened! ",
    "You have to see this to believe it! This is absolutely disgraceful! ",
    "Oh boy, it gets worse! Look at what this corrupt establishment did! ",
    "This is the shocking news that the fake media is refusing to talk about! "
]

CLICKBAIT_BODY_SUFFIXES = [
    " This is typical of the corrupt establishment! Share this everywhere!",
    " No one is talking about this in the mainstream media! They want to keep us in the dark!",
    " What do you think about this? Let us know in the comments! This is absolutely crazy!",
    " This is an absolute disaster! Share if you agree!",
    " Spread the word! We must make sure the truth gets out!"
]

def sensationalize_article(title: str, text: str) -> tuple[str, str]:
    """Decorate a real, dry wire article with sensational blog-style language."""
    # Prefix title
    title_prefix = random.choice(CLICKBAIT_HEADLINE_PREFIXES)
    new_title = f"{title_prefix}{title}"
    if random.random() > 0.5:
        new_title = new_title.upper()

    # Decorate body text
    body_prefix = random.choice(CLICKBAIT_BODY_PREFIXES)
    body_suffix = random.choice(CLICKBAIT_BODY_SUFFIXES)

    # Randomly capitalize emotional words in body
    words_to_caps = ["embarrassing", "disaster", "secret", "scandal", "shocking", "corrupt", "lies", "truth"]
    new_text = f"{body_prefix}{text}{body_suffix}"

    for word in words_to_caps:
        if random.random() > 0.3:
            new_text = new_text.replace(f" {word} ", f" {word.upper()} ")
            new_text = new_text.replace(f" {word}s ", f" {word.upper()}S ")

    return new_title, new_text

def generate_sensational_reals(real_df: pd.DataFrame, num_samples: int = 400, seed: int = 42) -> pd.DataFrame:
    """Take a subset of real articles and add highly emotional/sensational clickbait phrasing.

    Args:
        real_df: DataFrame containing real news (from True.csv).
        num_samples: Number of articles to augment.
        seed: Random seed for reproducibility.

    Returns:
        pd.DataFrame with columns: title, text, label (1 = REAL), subject ('politics')
    """
    random.seed(seed)

    # Sample from real_df
    sample_df = real_df.sample(n=min(num_samples, len(real_df)), random_state=seed).copy()

    augmented_data = []
    for _, row in sample_df.iterrows():
        title = str(row.get('title', ''))
        text = str(row.get('text', ''))

        new_title, new_text = sensationalize_article(title, text)

        augmented_data.append({
            "title": new_title,
            "text": new_text,
            "label": 1,  # REAL
            "subject": row.get('subject', 'politics')
        })

    logger.info("Generated %d Sensational Real articles.", len(augmented_data))
    return pd.DataFrame(augmented_data)
