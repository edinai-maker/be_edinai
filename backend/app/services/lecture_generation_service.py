"""
Lecture Generation Service - Core AI/LLM Integration
Handles all lecture content generation, math detection, and prompting
"""

import asyncio
import json
import re
from typing import Any, Dict, List, Optional, Tuple
from groq import Groq


# ============================================================================
# MATH UTILITIES
# ============================================================================

def wrap_math_expressions(text: str) -> str:
    """Wrap mathematical expressions in LaTeX delimiters for proper rendering."""
    
    math_patterns = [
        (r'\b(\d+\s*[+\-×÷]\s*\d+)\b', r'$\1$'),
        (r'\b(\d+\s*[+\-×÷]\s*\d+\s*[+\-×÷]\s*\d+)\b', r'$\1$'),
        (r'\b(\d+)/(\d+)\b', r'$\\frac{\1}{\2}$'),
        (r'\b(\w+)\^(\w+)\b', r'$\1^{\2}$'),
        (r'\b(\w+)_(\w+)\b', r'$\1_{\2}$'),
        (r'\b(a\s*×\s*b\s*=\s*HCF\(a,b\)\s*×\s*LCM\(a,b\))', r'$$\1$$'),
        (r'\b(HCF|LCM|GCD|sin|cos|tan|log|ln|exp)\(', r'$\\\1('),
        (r'\b([a-zA-Z]\s*[=≠<>≤≥]\s*[\d\w+\-×÷^()√π]+)', r'$\1$'),
    ]
    narration = re.sub(r'(?<!\\\)(\d+/\d+)', r'$\\frac{\1}$', narration)
    narration = re.sub(r'\^(\d+)', r'^{\1}', narration)
    narration = re.sub(r'(=|\+|-|×|÷)\s*(\d+)', r'\1 $\2$', narration)
    return narration
    
    for pattern, replacement in math_patterns:
        text = re.sub(pattern, replacement, text)
    
    text = text.replace('%', '\\%').replace('&', '\\&')
    return text


def detect_math_content(text: str) -> bool:
    """Enhanced math content detection with better accuracy."""
    
    # Expanded math keywords
    math_keywords = [
        # English - Core Math
        'mathematics', 'math', 'maths', 'algebra', 'arithmetic', 'geometry', 'trigonometry',
        'polynomial', 'quadratic', 'linear', 'equation', 'formula', 'theorem', 'proof',
        'prime', 'composite', 'factor', 'factorization', 'factorisation', 'divisor',
        'hcf', 'lcm', 'gcd', 'fraction', 'decimal', 'percentage', 'ratio', 'proportion',
        'exponent', 'power', 'square', 'cube', 'root', 'logarithm', 'exponential',
        'derivative', 'integral', 'calculus', 'differentiation', 'integration',
        'angle', 'triangle', 'circle', 'rectangle', 'area', 'perimeter', 'volume',
        'coordinate', 'graph', 'function', 'domain', 'range', 'slope', 'intercept',
        'matrix', 'determinant', 'vector', 'scalar', 'sequence', 'series', 'progression',
        'probability', 'statistics', 'mean', 'median', 'mode', 'variance', 'deviation',
        
        # Hindi - गणित keywords
        'गणित', 'गणितीय', 'बीजगणित', 'अंकगणित', 'ज्यामिति', 'त्रिकोणमिति',
        'समीकरण', 'सूत्र', 'प्रमेय', 'उपपत्ति', 'सिद्ध', 'प्रमाण', 'साबित',
        'अभाज्य', 'भाज्य', 'गुणनखंड', 'गुणनखण्डन', 'भाजक', 'विभाजक',
        'महत्तम', 'समापवर्तक', 'लघुत्तम', 'समापवर्त्य', 'भिन्न', 'दशमलव',
        'प्रतिशत', 'अनुपात', 'समानुपात', 'घात', 'घातांक', 'वर्ग', 'घन',
        'मूल', 'वर्गमूल', 'घनमूल', 'लघुगणक', 'कोण', 'त्रिभुज', 'वृत्त',
        'आयत', 'क्षेत्रफल', 'परिमाप', 'आयतन', 'निर्देशांक', 'ग्राफ', 'फलन',
        'आव्यूह', 'सारणिक', 'सदिश', 'अदिश', 'अनुक्रम', 'श्रेणी', 'प्रगति',
        'प्रायिकता', 'सांख्यिकी', 'माध्य', 'माध्यिका', 'बहुलक',
        'यूक्लिड', 'एल्गोरिदम', 'एल्गोरिथम', 'विभाजन', 'पूर्णांक',
        'वास्तविक', 'संख्या', 'परिमेय', 'अपरिमेय',
        
        # Gujarati - ગણિત keywords
        'ગણિત', 'ગાણિતિક', 'બીજગણિત', 'અંકગણિત', 'ભૂમિતિ', 'ત્રિકોણમિતિ',
        'સમીકરણ', 'સૂત્ર', 'પ્રમેય', 'પુરાવો', 'સાબિતી',
        'અવિભાજ્ય', 'ભાજ્ય', 'અવયવ', 'ગુણાકાર', 'ભાગાકાર', 'ભાજક',
        'મહત્તમ', 'લઘુત્તમ', 'સમાપવર્તક', 'ભિન્ન', 'દશાંશ',
        'ટકા', 'ગુણોત્તર', 'પ્રમાણ', 'ઘાત', 'ઘાતાંક', 'વર્ગ', 'ઘન',
        'મૂળ', 'વર્ગમૂળ', 'લોગરિધમ', 'કોણ', 'ત્રિકોણ', 'વર્તુળ',
        'લંબચોરસ', 'ક્ષેત્રફળ', 'પરિમિતિ', 'ઘનફળ', 'આલેખ', 'વિધેય',
        'અનુક્રમણિકા', 'શ્રેણી', 'સંભાવના', 'આંકડાશાસ્ત્ર', 'સરેરાશ',
    ]
    
    text_lower = text.lower()
    
    # Count keyword matches
    keyword_count = sum(1 for kw in math_keywords if kw.lower() in text_lower)
    
    # Strong math patterns (more comprehensive)
    strong_patterns = [
        r'(theorem|proof|lemma|corollary|axiom)',
        r'(प्रमेय|सिद्ध|प्रमाण|उपपत्ति|साबित)',
        r'(પ્રમેય|સાબિતી|પુરાવો)',
        r'(equation|formula|expression)',
        r'(समीकरण|सूत्र|व्यंजक)',
        r'(સમીકરણ|સૂત્ર)',
        r'(hcf|lcm|gcd|gcf)',
        r'(महत्तम|समापवर्तक|लघुत्तम)',
        r'(મહત્તમ|લઘુત્તમ|સમાપવર્તક)',
        r'(prime|factor|divisor|multiple)',
        r'(अभाज्य|गुणनखंड|भाजक)',
        r'(અવિભાજ્ય|અવયવ|ભાજક)',
        r'(solve|find|calculate|prove|verify)',
        r'(हल करें|ज्ञात करें|सिद्ध करें)',
        r'(ઉકેલો|શોધો|સાબિત કરો)',
        r'(rational|irrational|real number)',
        r'(परिमेय|अपरिमेय|वास्तविक संख्या)',
        r'(યુક્લિડ|વિભાજન|અલ્ગોરિધમ)',
        r'(यूक्लिड|विभाजन|एल्गोरिदम|एल्गोरिथम)',
        r'(euclid|division|algorithm)',
        r'(fundamental theorem)',
        r'(आधारभूत प्रमेय|मूलभूत प्रमेय)',
    ]
    
    pattern_matches = sum(1 for p in strong_patterns if re.search(p, text_lower))
    
    # Math symbols and notations
    math_symbols = ['×', '÷', '√', '²', '³', '∞', '≠', '≤', '≥', '±', '∑', '∏', 'π']
    symbol_count = sum(text.count(s) for s in math_symbols)
    
    # Basic operators (more lenient counting)
    basic_ops = ['+', '-', '=']
    basic_count = sum(text.count(s) for s in basic_ops)
    
    # Math-specific number patterns
    number_patterns = [
        r'\d+\s*[+\-×÷]\s*\d+',  # 5 + 3, 10 × 2
        r'\d+\^\d+',              # 2^3
        r'\d+/\d+',               # 3/4 (fraction)
        r'\b\d+\.\d+\b',          # 3.14 (decimal)
    ]
    number_pattern_count = sum(1 for p in number_patterns if re.search(p, text))
    
    # Debug information
    print(f"🔍 Math Detection Debug:")
    print(f"   Keywords: {keyword_count}")
    print(f"   Patterns: {pattern_matches}")
    print(f"   Symbols: {symbol_count}")
    print(f"   Basic ops: {basic_count}")
    print(f"   Number patterns: {number_pattern_count}")
    
    # More lenient detection criteria
    return (
        keyword_count >= 2 or           # Just 2 math keywords
        pattern_matches >= 1 or         # Any strong pattern match
        symbol_count >= 3 or            # 3+ math symbols
        number_pattern_count >= 3 or    # 3+ math number patterns
        (keyword_count >= 1 and basic_count >= 5)  # 1 keyword + 5 operators
    )

def detect_science_content(text: str) -> bool:
    """Detect if content is Science-related."""
    science_keywords = [
        "physics", "chemistry", "biology", "botany", "zoology",
        "electricity", "magnetism", "cell", "photosynthesis",
        "ecosystem", "force", "motion", "energy", "atom", "molecule",
        "reaction", "experiment", "lab", "genetics",
        "विज्ञान", "भौतिकी", "रसायन", "जीवविज्ञान",
        "વિજ્ઞાન", "ભૌતિક", "રસાયણ", "જીવવિજ્ઞાન"
    ]
    text_lower = text.lower()
    keyword_hits = sum(1 for keyword in science_keywords if keyword in text_lower)
    return keyword_hits >= 4


# ============================================================================
# PROMPT TEMPLATES
# ============================================================================


def _get_word_targets(duration: int) -> Dict[str, str]:
    """Return intro/normal/deep word targets for supported durations."""
    if duration == 30:
        return {"intro": "180-220", "normal": "260-320", "deep": "420-520"}
    if duration == 45:
        return {"intro": "320-420", "normal": "450-550", "deep": "750-900"}
    return {"intro": "420-520", "normal": "580-680", "deep": "900-1100"}


def create_lecture_prompt(*, text: str, language: str, duration: int, style: str) -> str:
    """Universal prompt for all subjects with subject-specific enhancements."""
    
    # Duration-based word counts
    words = _get_word_targets(duration)
    
    # Language instructions
    language_instructions = {
        "Gujarati": (
            "LANGUAGE RULES:\n"
            "1. Technical terms ONLY in English.\n"
            "2. All explanations in simple Gujarati for std 5-12.\n"
            "3. Use short, clear sentences.\n"
            "4. Keep tone friendly and student-like.\n"
            "5. If the chapter text is in another language/script, translate it into Gujarati before using it.\n"
        ),
        "Hindi": (
            "LANGUAGE RULES:\n"
            "1. Technical terms ONLY in English.\n"
            "2. All explanations in simple Hindi for std 5-12.\n"
            "3. Use short, clear sentences.\n"
            "4. Keep tone friendly and engaging.\n"
            "5. If the chapter text is in another language/script, translate it into Hindi before using it.\n"
        ),
        "English": (
            "LANGUAGE RULES:\n"
            "1. Use simple, student-friendly English.\n"
            "2. Clear explanations for std 5-12.\n"
            "3. Keep tone engaging and easy to understand.\n"
            "4. Translate any non-English source material into English while keeping meaning intact.\n"
        )
    }
    
    lang_instruction = language_instructions.get(language, f"Generate in {language}")
    
    language_enforcement_rules = {
        "Hindi": (
            "LANGUAGE ENFORCEMENT (Hindi):\n"
            "1. Titles, bullets, narration, subnarrations, and questions must be entirely in Hindi (Devanagari script).\n"
            "2. Translate Gujarati or any other script into Hindi; never copy-source sentences verbatim unless already Hindi.\n"
            "3. Only keep universally accepted technical terms in English (e.g., DNA, photosynthesis).\n"
            "4. Never mix Gujarati script or Romanized Hindi in the output.\n"
        ),
        "Gujarati": (
            "LANGUAGE ENFORCEMENT (Gujarati):\n"
            "1. Titles, bullets, narration, subnarrations, and questions must be entirely in Gujarati script.\n"
            "2. Translate Hindi or any other script into Gujarati before writing.\n"
            "3. Only retain globally accepted English technical terms when unavoidable.\n"
            "4. Never include Devanagari script or long English sentences in the narration.\n"
        ),
        "English": (
            "LANGUAGE ENFORCEMENT (English):\n"
            "1. Write every title, narration, bullet, question, and summary in fluent English.\n"
            "2. Translate Indian language passages into English rather than quoting them.\n"
            "3. Maintain the same structure and detail while paraphrasing the source.\n"
        ),
    }
    default_enforcement = (
        f"LANGUAGE ENFORCEMENT ({language}):\n"
        f"1. Use {language} consistently for titles, narration, bullets, and questions.\n"
        f"2. Translate source text into {language} instead of copying foreign-language sentences.\n"
        "3. Keep structure identical to the requested template.\n"
    )
    language_enforcement = language_enforcement_rules.get(language, default_enforcement)
    
    # Detect subject type
    is_math = detect_math_content(text)
    is_science = detect_science_content(text)
    
    print(f"🔍 Detected: Math={is_math}, Science={is_science}")
    
    # Subject-specific instructions
    subject_instructions = []
    
    if is_math:
        subject_instructions.append(
            "📐 MATH CONTENT DETECTED - SPECIAL REQUIREMENTS:\n"
            "1. Format ALL mathematical expressions with LaTeX:\n"
            "   - Inline: $x^2 + 5x + 6$\n"
            "   - Display: $$x = \\frac{-b \\pm \\sqrt{b^2-4ac}}{2a}$$\n"
            "   - Fractions: \\frac{numerator}{denominator}\n"
            "   - Exponents: a^{b}, Subscripts: a_{b}\n"
            "2. Extract ACTUAL problems from source (NO generic examples)\n"
            "3. Show COMPLETE step-by-step solutions:\n"
            "   - Given: [State problem]\n"
            "   - To Find: [What to solve]\n"
            "   - Step 1: [First step with LaTeX]\n"
            "     $$equation_1$$\n"
            "   - Step 2: [Next step]\n"
            "     $$equation_2$$\n"
            "   - Continue all steps...\n"
            "   - Final Answer: $$\\boxed{result}$$\n"
            "4. For theorems: State theorem, then prove step-by-step\n"
            "5. Extract formulas from source and explain when to use them\n"
            "6. Slides 4-7 MUST include at least ONE fully worked problem each\n"
        )
    
    if is_science:
        subject_instructions.append(
            "🔬 SCIENCE CONTENT DETECTED:\n"
            "1. Mention key formulas, laws, and scientific principles\n"
            "2. Describe experiments and observations\n"
            "3. Use diagrams and label key parts (describe verbally)\n"
            "4. Connect theory to real-life applications\n"
            "5. Include scientific notation where relevant\n"
        )
    
    if not subject_instructions:
        subject_instructions.append(
            "📚 GENERAL CONTENT:\n"
            "1. Use practical examples from the chapter\n"
            "2. Keep explanations clear and relatable\n"
            "3. Include real-world connections\n"
        )
    
    subject_block = "\n\n".join(subject_instructions)
    
    prompt = f"""
Create a {duration}-minute educational lecture from the provided chapter.

{lang_instruction}

{language_enforcement}

{subject_block}

CRITICAL INSTRUCTIONS:
1. Read ALL topics and subtopics from the source material
2. Generate content ONLY from the actual chapter content
3. Extract actual examples, problems, and concepts from source
4. Maintain the exact sequence of topics as they appear
5. NO invented/generic examples - use source material only

OUTPUT FORMAT:
Return ONLY valid JSON with exactly 9 slides:
{{
  "slides": [9 slide objects],
  "estimated_duration": {duration}
}}

MANDATORY 9-SLIDE STRUCTURE:

=== SLIDE 1: INTRODUCTION ===
{{
  "title": "Introduction to [Actual Topic from Source]",
  "bullets": [],
  "narration": "Start with interesting hook. Explain the topic and its importance. Preview what students will learn. ({words['intro']} words)",
  "question": ""
}}

=== SLIDE 2: KEY CONCEPTS ===
{{
  "title": "Important Concepts You Must Know",
  "bullets": ["Concept 1", "Concept 2", "Concept 3"],
  "narration": "Explain key concepts from the chapter with short examples. ({words['normal']} words)",
  "question": ""
}}

=== SLIDE 3: DEEP UNDERSTANDING ===
{{
  "title": "Deep Understanding of the Topic",
  "bullets": [],
  "narration": "Explain deeper meaning behind the topic using clear steps. ({words['normal']} words)",
  "question": ""
}}

=== SLIDES 4-7: DETAILED TEACHING ===
For EACH slide:
- Pick next subtopic from source (in order)
- Explain the concept clearly
- Provide examples/applications from source
- If Math: Include fully worked problem with step-by-step solution
- If Science: Describe relevant experiment or application
- Target: {words['deep']} words

{{
  "title": "[Subtopic from Source]",
  "bullets": [],
  "narration": "[Detailed explanation with examples from source]",
  "subnarrations": [
    {{
      "title": "[Subtopic name]",
      "summary": "[40-60 word summary of key points]"
    }}
  ],
  "question": ""
}}

=== SLIDE 8: PRACTICAL APPLICATIONS ===
{{
  "title": "Where This Knowledge Is Useful",
  "bullets": [],
  "narration": "Explain real-life applications and practical uses. ({words['normal']} words)",
  "question": ""
}}

=== SLIDE 9: QUIZ & SUMMARY ===
{{
  "title": "Quick Summary & Quiz",
  "bullets": [],
  "narration": "Summarize key concepts in 3-5 lines.",
  "question": "1. Question 1?\\n2. Question 2?\\n3. Question 3?\\n4. Question 4?\\n5. Question 5?"
}}

SOURCE MATERIAL (Use ALL content):
{text[:60000]}

Generate exactly 9 slides based entirely on the source content above.
Return ONLY valid JSON, NO markdown fences.
"""
    
    return prompt


# ============================================================================
# GROQ SERVICE - MAIN GENERATION ENGINE
# ============================================================================

class GroqService:
    """Service wrapper around the Groq chat completion API."""
    
    def __init__(self, api_key: str) -> None:
        self._client: Optional[Groq] = Groq(api_key=api_key) if api_key else None
        self._rewrite_cache: Dict[Tuple[str, str], str] = {}
    
    @property
    def configured(self) -> bool:
        return self._client is not None
    
    def _format_source_material(self, text: str) -> str:
        """Extract and format complete topic content including all narrations."""
        
        if "Topic:" not in text:
            return text
        
        formatted_sections = []
        lines = text.split('\n')
        i = 0
        
        while i < len(lines):
            line = lines[i].strip()
            
            if line.startswith("Topic:"):
                topic_title = line.replace("Topic:", "").strip()
                formatted_sections.append(f"\n{'='*60}")
                formatted_sections.append(f"TOPIC: {topic_title}")
                formatted_sections.append(f"{'='*60}\n")
                i += 1
                
            elif line.startswith("Subtopic:"):
                subtopic_line = line.replace("Subtopic:", "").strip()
                i += 1
                
                # Next line(s) might contain narration in quotes
                narration = ""
                while i < len(lines):
                    next_line = lines[i].strip()
                    if next_line.startswith("Topic:") or next_line.startswith("Subtopic:"):
                        break
                    if next_line:
                        # Remove quotes and collect narration
                        cleaned = next_line.strip('"').strip()
                        if cleaned:
                            narration += " " + cleaned
                    i += 1
                
                formatted_sections.append(f"\n### {subtopic_line}")
                if narration:
                    formatted_sections.append(f"{narration.strip()}\n")
                
            else:
                i += 1
        
        result = "\n".join(formatted_sections)
        
        # If formatting produced good content, use it
        if len(result.strip()) > len(text.strip()) * 0.5:
            print(f"\n📝 Formatted source material ({len(result)} chars):")
            print(result[:500] + "..." if len(result) > 500 else result)
            return result
        
        # Otherwise return original
        return text
    async def generate_lecture_content(
        self, 
        text: str, 
        language: str = "Hindi", 
        duration: int = 30, 
        style: str = "clear"
    ) -> Dict[str, Any]:
        """Generate lecture with automatic subject detection."""
        
        if not self._client:
            raise RuntimeError("Groq API key missing")
        
        # Detect subject type
        is_math = detect_math_content(text)
        is_science = detect_science_content(text)
        
        print(f"📚 Generating lecture: Math={is_math}, Science={is_science}, Lang={language}")
        
        # Use universal prompt (with subject-specific enhancements)
        prompt = create_lecture_prompt(
            text=text,
            language=language,
            duration=duration,
            style=style
        )
        
        # Call API
        completion = await asyncio.to_thread(
            self._client.chat.completions.create,
            messages=[
                {
                    "role": "system",
                    "content": f"You are an expert {language} teacher. Output ONLY valid JSON with exactly 9 slides."
                },
                {"role": "user", "content": prompt}
            ],
            model="openai/gpt-oss-120b",
            temperature=0.2 if is_math else 0.3,
            max_tokens=32000
        )
        
        response = completion.choices[0].message.content.strip()
        response = re.sub(r"^```(?:json)?\s*|\s*```$", "", response, flags=re.DOTALL).strip()
        
        try:
            data = json.loads(response)
        except json.JSONDecodeError as e:
            repaired_response = self._repair_json_response(response)
            try:
                data = json.loads(repaired_response)
            except json.JSONDecodeError as repaired_error:
                raise RuntimeError(f"Invalid JSON from model: {repaired_error}") from e
        
        slides = data.get("slides", [])
        fallback_used = False
        if len(slides) != 9:
            try:
                slides = await self._retry_exact_slide_count(
                    base_prompt=prompt,
                    original_response=response,
                    language=language,
                    duration=duration,
                    is_math=is_math,
                )
            except RuntimeError:
                fallback_used = True
                slides = self._coerce_slide_count_to_nine(
                    slides=slides,
                    source_text=text,
                    language=language,
                    duration=duration,
                )
        
        # Post-process based on subject
        if is_math:
            slides = self._enhance_math_slides(slides)
        
        # Add subnarrations for slides 4-7
        for i in range(3, 7):
            if i < len(slides):
                slide = slides[i]
                if not slide.get("subnarrations"):
                    narration = slide.get("narration", "")
                    words = narration.split()
                    summary = " ".join(words[:50]) + ("..." if len(words) > 50 else "")
                    slide["subnarrations"] = [{
                        "title": slide.get("title", ""),
                        "summary": summary
                    }]
            
        # Ensure minimum narration lengths before estimating duration
        self._enforce_minimum_narration(
            slides,
            language,
            requested_duration=duration,
        )
        
        await self._enforce_language_output(slides, language)
        estimated_duration = self._estimate_duration_minutes(
            slides,
            requested_duration=duration,
            language=language,
        )
        
        return {
            "slides": slides,
            "estimated_duration": estimated_duration,
            "fallback_used": fallback_used,
        }
    async def _retry_exact_slide_count(
        self,
        *,
        base_prompt: str,
        original_response: str,
        language: str,
        duration: int,
        is_math: bool,
    ) -> List[Dict[str, Any]]:
        """Retry once with explicit instruction to produce exactly 9 slides."""
        if not self._client:
            raise RuntimeError("Groq API key missing")
        
        retry_messages = [
            {
                "role": "system",
                "content": f"You are an expert {language} teacher. Output ONLY valid JSON with exactly 9 slides.",
            },
            {"role": "user", "content": base_prompt},
            {"role": "assistant", "content": original_response},
            {
                "role": "user",
                "content": (
                    "Your previous JSON response did not contain exactly 9 slides. "
                    "Regenerate the lecture now, strictly following the 9-slide template. "
                    "Return ONLY valid JSON with exactly 9 slide objects inside the `slides` array "
                    "and include the `estimated_duration` field."
                ),
            },
        ]
        
        completion = await asyncio.to_thread(
            self._client.chat.completions.create,
            messages=retry_messages,
            model="openai/gpt-oss-120b",
            temperature=0.2 if is_math else 0.3,
            max_tokens=32000,
        )
        
        retry_response = completion.choices[0].message.content.strip()
        retry_response = re.sub(r"^```(?:json)?\s*|\s*```$", "", retry_response, flags=re.DOTALL).strip()
        
        try:
            data = json.loads(retry_response)
        except json.JSONDecodeError:
            repaired_response = self._escape_invalid_backslashes(retry_response)
            data = json.loads(repaired_response)
        
        slides = data.get("slides", [])
        if len(slides) != 9:
            raise RuntimeError(f"Expected 9 slides after retry, got {len(slides)}")
        
        return slides
    def _repair_json_response(self, text: str) -> str:
        """Attempt to fix malformed JSON produced by the LLM."""
        cleaned = text.strip()
        if not cleaned:
            return cleaned
        
        # Remove surrounding markdown fences if any
        cleaned = re.sub(r'^```(?:json)?\s*|\s*```$', '', cleaned, flags=re.IGNORECASE)
        
        cleaned = self._escape_invalid_backslashes(cleaned)
        cleaned = self._ensure_valid_unicode_sequences(cleaned)
        return cleaned
    def _escape_invalid_backslashes(self, text: str) -> str:
        """Escape stray backslashes (e.g. from LaTeX) to keep JSON valid."""
        builder: List[str] = []
        i = 0
        length = len(text)
        while i < length:
            ch = text[i]
            if ch != '\\':
                builder.append(ch)
                i += 1
                continue
            
            # At this point ch is backslash
            if i + 1 >= length:
                builder.append('\\\\')
                i += 1
                continue
            
            nxt = text[i + 1]
            if nxt in '"\\/bfnrt':
                builder.append('\\' + nxt)
                i += 2
                continue
            
            if nxt == 'u':
                hex_part = text[i + 2 : i + 6]
                if len(hex_part) == 4 and all(c in '0123456789abcdefABCDEF' for c in hex_part):
                    builder.append('\\u' + hex_part)
                    i += 6
                    continue
                builder.append('\\\\u')
                i += 2
                continue
            
            builder.append('\\\\')
            i += 1
            continue
        
        return ''.join(builder)
    def _ensure_valid_unicode_sequences(self, text: str) -> str:
        """Ensure control characters/newlines are properly escaped."""
        # Replace raw carriage returns / tabs with escaped versions
        text = text.replace('\r', '\\r').replace('\t', '\\t')
        
        # Ensure newlines inside strings are escaped
        text = text.replace('\n', '\\n')
        return text
    def _enhance_math_slides(self, slides: List[Dict]) -> List[Dict]:
        """Post-process math slides to ensure proper LaTeX formatting."""
        
        for slide in slides:
            narration = slide.get("narration", "")
            
            # Only enhance if it contains math expressions
            if any(c in narration for c in ['=', '+', '-', '×', '÷', '^', '/']):
                
                # Fix fractions: 5/3 -> $\frac{5}{3}$
                narration = re.sub(
                    r'(?<![\\$\w])(\d+)/(\d+)(?![\\$\w])', 
                    r'$\\frac{\1}{\2}$', 
                    narration
                )
                
                # Fix exponents: x^2 -> x^{2}
                narration = re.sub(r'\^(\d+)', r'^{\1}', narration)
                
                # Wrap standalone equations if not already wrapped
                if '$' not in narration[:50]:  # Check if no $ in beginning
                    narration = re.sub(
                        r'\b([a-zA-Z])\s*=\s*([^.\n,]+)',
                        r'$\1 = \2$',
                        narration
                    )
                
                slide["narration"] = narration
            
            # Process bullets
            bullets = slide.get("bullets", [])
            processed_bullets = []
            for bullet in bullets:
                if isinstance(bullet, str):
                    # Add $ wrapping if contains = and no $ already
                    if '=' in bullet and '$' not in bullet:
                        bullet = f"${bullet}$"
                    processed_bullets.append(bullet)
            
            if processed_bullets:
                slide["bullets"] = processed_bullets
        
        return slides

    def _coerce_slide_count_to_nine(
        self,
        *,
        slides: List[Dict[str, Any]],
        source_text: str,
        language: str,
        duration: int,
    ) -> List[Dict[str, Any]]:
        """Trim or synthesize slides so the payload always contains exactly 9 entries."""
        slides = slides or []
        normalized: List[Dict[str, Any]] = []
        for idx, slide in enumerate(slides[:9]):
            normalized.append(
                {
                    "number": idx + 1,
                    "title": str(slide.get("title") or f"Slide {idx + 1}").strip(),
                    "bullets": slide.get("bullets") if isinstance(slide.get("bullets"), list) else [],
                    "narration": str(slide.get("narration") or "").strip(),
                    "question": str(slide.get("question") or "").strip(),
                }
            )
        
        if len(normalized) == 9:
            return normalized
        
        topic_hint = self._guess_topic_title(source_text)
        sections = self._split_source_into_sections(source_text, 9)
        blueprints = self._slide_blueprints(topic_hint, duration)
        
        while len(normalized) < 9:
            idx = len(normalized)
            blueprint = blueprints[idx]
            narration = self._summarize_text(
                text=sections[idx],
                target_words=blueprint["target_words"],
                fallback=blueprint["narration"],
            )
            placeholder = {
                "number": idx + 1,
                "title": blueprint["title"],
                "bullets": blueprint["bullets"],
                "narration": narration,
                "question": blueprint["question"],
            }
            normalized.append(placeholder)
        
        return normalized

    def _slide_blueprints(self, topic: str, duration: int) -> List[Dict[str, Any]]:
        """Provide deterministic structure for each slide slot."""
        word_targets = _get_word_targets(duration)
        detailed_words = word_targets["deep"]
        normal_words = word_targets["normal"]
        
        detailed_template = {
            "bullets": [],
            "question": "",
            "target_words": detailed_words,
        }
        
        return [
            {
                "title": f"Introduction to {topic}",
                "bullets": [],
                "narration": f"Introduce the chapter on {topic} and outline learning goals.",
                "question": "",
                "target_words": word_targets["intro"],
            },
            {
                "title": "Key Concepts Overview",
                "bullets": [f"Core idea about {topic}", "Important fact", "Key relationship"],
                "narration": f"Summarize the essential concepts students must know about {topic}.",
                "question": "",
                "target_words": normal_words,
            },
            {
                "title": "Deep Understanding",
                "bullets": [],
                "narration": f"Explain why these ideas matter and how they connect within {topic}.",
                "question": "",
                "target_words": normal_words,
            },
            *[
                {
                    **detailed_template,
                    "title": f"Detailed Teaching {i - 3}",
                    "narration": f"Explain subtopic {i - 3} from {topic} with worked steps.",
                }
                for i in range(4, 8)
            ],
            {
                "title": "Practical Applications",
                "bullets": [],
                "narration": f"Describe how {topic} shows up in real life, labs, or exams.",
                "question": "",
                "target_words": normal_words,
            },
            {
                "title": "Quiz & Summary",
                "bullets": [],
                "narration": "Summarize the lecture and prepare students for recall.",
                "question": "1. Question 1?\n2. Question 2?\n3. Question 3?\n4. Question 4?\n5. Question 5?",
                "target_words": "120-150",
            },
        ]

    def _split_source_into_sections(self, text: str, parts: int) -> List[str]:
        """Split source text into roughly even segments for fallback synthesis."""
        cleaned = re.sub(r"\s+", " ", text or "").strip()
        if not cleaned:
            return [""] * parts
        
        sentences = re.split(r"(?<=[.!?।])\s+", cleaned)
        if not sentences:
            return [""] * parts
        
        chunk_size = max(1, len(sentences) // parts)
        sections: List[str] = []
        for i in range(parts):
            start = i * chunk_size
            end = (i + 1) * chunk_size if i < parts - 1 else len(sentences)
            chunk = " ".join(sentences[start:end]).strip()
            sections.append(chunk)
        
        while len(sections) < parts:
            sections.append("")
        
        return sections[:parts]

    def _summarize_text(self, text: str, target_words: str, *, fallback: str) -> str:
        """Summarize source text to the lower bound of the requested word range."""
        if not text:
            return fallback
        
        match = re.match(r"(\d+)", target_words or "")
        limit = int(match.group(1)) if match else 120
        words = text.split()
        if not words:
            return fallback
        
        snippet = " ".join(words[:limit])
        return snippet + ("..." if len(words) > limit else "")

    def _guess_topic_title(self, text: str) -> str:
        """Heuristically derive a title-sized phrase from the source text."""
        for line in text.splitlines():
            candidate = line.strip()
            if len(candidate.split()) >= 2:
                return candidate[:80]
        return "This Chapter"
    
    async def answer_question(
        self,
        *,
        question: str,
        context: str,
        language: str = "English",
        script_instruction: str | None = None,
        question_language_hint: str | None = None,
        answer_type: str | None = None,
        is_edit_command: bool = False,
    ) -> str | Dict[str, str] | Dict[str, Any]:
        """Answer questions or handle edit commands."""
        if not self.configured:
            return "Please review the lecture materials."
        
        system_prompt = f"You are a teaching assistant. Answer in {language}. Be brief and clear."
        user_prompt = f"Context: {context[:1000]}\n\nQuestion: {question}\n\nAnswer:"
        
        try:
            completion = await asyncio.to_thread(
                self._client.chat.completions.create,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                model="llama-3.3-70b-versatile",
                temperature=0.3,
                max_tokens=5000
            )
            
            return completion.choices[0].message.content
        except Exception:
            return "I'm having trouble processing your question."
    
    async def _handle_edit_command(self, question: str, context: str) -> Dict[str, Any]:
        """Handle edit commands for slide content."""
        system_prompt = (
            "You are editing lecture slides. "
            "Return JSON with 'edited_content' and 'explanation' fields."
        )
        user_prompt = f"Current content:\n{context}\n\nEdit: {question}"
        
        try:
            completion = await self._create_chat_completion(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                model="llama-3.3-70b-versatile",
                temperature=0.2,
            )
            
            response = completion["choices"][0]["message"]["content"]
            
            try:
                parsed = json.loads(response)
            except json.JSONDecodeError:
                match = re.search(r"\{[\s\S]*\}", response)
                if match:
                    parsed = json.loads(match.group(0))
                else:
                    parsed = None
            
            if parsed and isinstance(parsed, dict) and "edited_content" in parsed:
                return {
                    "answer": parsed.get("explanation", "Changes applied."),
                    "edited_content": parsed["edited_content"],
                }
            
            return {
                "answer": "Applied the edit.",
                "edited_content": response.strip(),
            }
            
        except Exception as e:
            return {"answer": f"Error: {str(e)}"}
    
    async def _handle_question(
        self, 
        question: str, 
        context: str, 
        language: str,
        script_instruction: Optional[str],
        question_language_hint: Optional[str],
        answer_type: Optional[str]
    ) -> str | Dict[str, Any]:
        """Handle regular Q&A."""
        system_prompt = (
            f"You are a teaching assistant for this {language} lecture. "
            "Provide brief, direct answers. Keep under 3 sentences."
        )
        if script_instruction:
            system_prompt += f" Follow: {script_instruction}"
        
        user_prompt = f"Context: {context[:1000]}\n\nQuestion: {question}\n\nAnswer:"
        
        try:
            completion = await self._create_chat_completion(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                model="llama-3.3-70b-versatile",
                max_tokens=5000,
            )
            content = completion["choices"][0]["message"]["content"]
            
            if answer_type == "json":
                try:
                    data = json.loads(content)
                    if isinstance(data, dict) and "display_text" in data:
                        data.setdefault("tts_text", data["display_text"])
                        return data
                except json.JSONDecodeError:
                    pass
            
            return content
        except Exception:
            return "I'm having trouble processing your question."
    
    async def _create_chat_completion(
        self,
        *,
        messages: List[Dict[str, str]],
        model: str,
        temperature: float = 0.3,
        max_tokens: int = 32000,
    ) -> Dict[str, Any]:
        """Create chat completion via Groq API."""
        if not self._client:
            raise RuntimeError("Groq client not configured")
        
        def _invoke() -> Dict[str, Any]:
            completion = self._client.chat.completions.create(
                messages=messages,
                model=model,
                temperature=temperature,
                max_tokens=max_tokens
            )
            return completion.to_dict() if hasattr(completion, "to_dict") else completion
        
        return await asyncio.to_thread(_invoke)
    
    def _create_lecture_prompt(
        self,
        *,
        text: str,
        language: str,
        duration: int,
        style: str,
    ) -> str:
        """Create prompt with subject-aware guidance embedded."""
        return create_lecture_prompt(
            text=text,
            language=language,
            duration=duration,
            style=style,
        )
    
    def _parse_lecture_response(self, response: str) -> Dict[str, Any]:
        """Parse AI response and extract slides."""
        response = response.strip()
        
        # Clean markdown
        if response.startswith("```"):
            response = re.sub(r'^```(?:json|JSON)?\s*\n?', '', response)
            response = re.sub(r'\n?```\s*$', '', response)
            response = response.strip()
        
        # Try JSON first
        parsed_json = self._try_parse_json(response)
        if parsed_json:
            return parsed_json
        
        # Fallback to text parsing
        return self._parse_text_format(response)
    
    def _try_parse_json(self, response: str) -> Optional[Dict[str, Any]]:
        """Attempt to parse response as JSON."""
        try:
            data = json.loads(response)
        except json.JSONDecodeError:
            return None
        
        if not isinstance(data, dict) or not isinstance(data.get("slides"), list):
            return None
        
        # normalized_slides = []
        normalized_slides: List[Dict[str, Any]] = []
        for raw_slide in data["slides"]:
            if not isinstance(raw_slide, dict):
                continue
            
            title = str(raw_slide.get("title", "")).strip()
            bullets = raw_slide.get("bullets") or []
            narration = str(raw_slide.get("narration", "")).strip()
            question = str(raw_slide.get("question", "")).strip()
            
            if not title:
                continue
            
            cleaned_bullets = [str(b).strip() for b in bullets if str(b).strip()]
           

            slide_number = len(normalized_slides) + 1
            allow_bullets = slide_number in (2, 3)
            bullet_points = cleaned_bullets[:3] or [title] if allow_bullets else []
            
            normalized_slides.append(
                {
                    "number": slide_number,
                    "title": title,
                    "bullets": bullet_points,
                    "narration": narration,
                    "question": question if slide_number == 9 else "",
                }
            )
        
        if not normalized_slides:
            return None
        
        estimated = data.get("estimated_duration")
        try:
            estimated_duration = int(estimated)
        except (TypeError, ValueError):
            estimated_duration = 0

        self._attach_subtopics_to_slides(normalized_slides)

        computed_estimate = self._estimate_duration_minutes(normalized_slides)
        
        return {
            "slides": normalized_slides,
            "total_slides": len(normalized_slides),
            "estimated_duration": max(estimated_duration, computed_estimate),
        }
    
    def _parse_text_format(self, response: str) -> Dict[str, Any]:
        """Fallback parser for text format."""
        # slides = []
        slides: List[Dict[str, Any]] = []
        pattern = re.compile(r"(?i)slide\s*(\d+)\s*[:：]?")
        matches = list(pattern.finditer(response))
        
        if not matches:
            return {"slides": [], "total_slides": 0, "estimated_duration": 0}
        
        for index, match in enumerate(matches):
            start = match.end()
            end = matches[index + 1].start() if index + 1 < len(matches) else len(response)
            block = response[start:end].strip()
            lines = block.splitlines()
            
            slide_data = {
                "number": int(match.group(1)) if match.group(1).isdigit() else len(slides) + 1,
                "title": "",
                "bullets": [],
                "narration": "",
                "question": "",
            }
            
            current_section: Optional[str] = None
            for raw_line in lines:
                value = raw_line.strip()
                if not value:
                    continue
                
                upper = value.upper()
                if upper.startswith("TITLE:"):
                    slide_data["title"] = value.split(":", 1)[1].strip()
                    current_section = "title"
                elif upper.startswith("BULLETS:"):
                    current_section = "bullets"
                elif upper.startswith("NARRATION:"):
                    slide_data["narration"] = value.split(":", 1)[1].strip()
                    current_section = "narration"
                elif upper.startswith("QUESTION:"):
                    slide_data["question"] = value.split(":", 1)[1].strip()
                    current_section = "question"
                elif current_section == "bullets":
                    cleaned = value.lstrip("-*•0123456789. ").strip()
                    if cleaned:
                        slide_data["bullets"].append(cleaned)
                elif current_section == "narration":
                    slide_data["narration"] = (slide_data["narration"] + " " + value).strip()
                elif current_section == "question":
                    slide_data["question"] = (slide_data["question"] + " " + value).strip()
            
            slide_number = len(slides) + 1
            allow_bullets = slide_number in (2, 3)
            if allow_bullets:
                if not slide_data["bullets"]:
                    slide_data["bullets"] = [slide_data["title"]]
            else:
                slide_data["bullets"] = []
            
            if slide_data["title"] and (slide_data["narration"] or slide_data["question"]):
                slide_data["number"] = slide_number
                if slide_number == 9:
                    slide_data["question"] = ""
                slides.append(slide_data)
        
        self._attach_subtopics_to_slides(slides)
        return {
            "slides": slides,
            "total_slides": len(slides),
            "estimated_duration": self._estimate_duration_minutes(slides),
        }

    def _estimate_duration_minutes(
        self,
        slides: List[Dict[str, Any]],
        *,
        requested_duration: Optional[int] = None,
        language: Optional[str] = None,
    ) -> int:
        """Estimate lecture playback duration based on textual content."""

        if not slides:
            return int(requested_duration or 0)

        total_words = 0
        for slide in slides:
            if not isinstance(slide, dict):
                continue

            sections: List[str] = []

            for field in ("title", "narration", "question"):
                value = slide.get(field)
                if isinstance(value, str) and value.strip():
                    sections.append(value)

            bullets = slide.get("bullets")
            if isinstance(bullets, list):
                for bullet in bullets:
                    if isinstance(bullet, str) and bullet.strip():
                        sections.append(bullet)

            subnarrations = slide.get("subnarrations")
            if isinstance(subnarrations, list):
                for sub in subnarrations:
                    if isinstance(sub, dict):
                        summary = sub.get("summary")
                        if isinstance(summary, str) and summary.strip():
                            sections.append(summary)

            for section in sections:
                total_words += len(section.split())

        average_wpm = self._resolve_average_wpm(language)
        estimated_minutes = int(round(total_words / average_wpm)) if total_words else 0

        # Ensure minimum of 1 minute per populated slide to avoid zero estimates for sparse content
        populated_slide_count = sum(
            1 for slide in slides if isinstance(slide, dict) and slide.get("narration")
        )
        estimated_minutes = max(estimated_minutes, populated_slide_count)

        if requested_duration:
            # Never exceed 20% above the requested duration to avoid unrealistic inflation
            lower_cap = int(round(requested_duration))
            upper_cap = int(round(requested_duration * 1.2))
            estimated_minutes = max(estimated_minutes, lower_cap)
            estimated_minutes = min(estimated_minutes, upper_cap)

        return estimated_minutes

    @staticmethod
    def _resolve_average_wpm(language: Optional[str]) -> int:
        """Return a reasonable words-per-minute rate for narration."""

        normalized = (language or "").strip().lower()
        if normalized == "english":
            return 135
        if normalized == "hindi":
            return 118
        if normalized == "gujarati":
            return 112
        return 130
    
    
    def _validate_language_mixing(self, parsed: Dict[str, Any], language: str) -> List[str]:
        """Validate Hindi/Gujarati content has proper English mixing and script usage."""
        errors = []
        
        forbidden_words = {
            "आर्टिफिशियल इंटेलिजेंस": "Artificial Intelligence",
            "મશીન લર્નિંગ": "Machine Learning",
        }
        script_rules = {
            "Hindi": {
                "forbidden": re.compile(r"[\u0A80-\u0AFF]"),    # Gujarati script
            },
            "Gujarati": {
                "forbidden": re.compile(r"[\u0900-\u097F]"),    # Devanagari script
            },
        }
        
        slides = parsed.get("slides", [])
        for idx, slide in enumerate(slides):
            number = slide.get("number") or idx + 1
            title = str(slide.get("title", ""))
            narration = str(slide.get("narration", ""))
            # question = str(slide.get("question", ""))
            # combined_text = " ".join(filter(None, [title, narration, question]))
            question = str(slide.get("question", ""))
            combined_text = " ".join(filter(None, [title, narration, question]))
            rules = script_rules.get(language)
            if rules and combined_text:
                forbidden_pattern = rules.get("forbidden")
                if forbidden_pattern and forbidden_pattern.search(combined_text):
                    errors.append(
                        f"Slide {idx + 1}: Contains characters from a different Indian script than {language}"
                    )
            
            for wrong, correct in forbidden_words.items():
                if wrong in title or wrong in narration:
                    errors.append(f"Slide {idx + 1}: Found '{wrong}' - should be '{correct}'")
            
            word_count = len(narration.split())
            min_words = self._get_minimum_word_requirement(
                number,
                requested_duration=requested_duration,
            )
            
            if number == 9 and not narration and slide.get("question"):
                continue
            
            if min_words and word_count < min_words:
                errors.append(
                    f"Slide {idx + 1}: Narration too short ({word_count} words, need {min_words}+)"
                )
        
        return errors

    async def _enforce_language_output(self, slides: List[Dict[str, Any]], language: str) -> None:
        """Ensure every slide field respects the requested language."""
        if not slides or not language:
            return
        
        for slide in slides:
            for field in ("title", "narration", "question"):
                value = slide.get(field)
                if isinstance(value, str) and value.strip():
                    if not self._is_language_compliant(value, language):
                        slide[field] = await self._rewrite_text_language(value, language)
            
            bullets = slide.get("bullets")
            if isinstance(bullets, list) and bullets:
                rewritten_bullets: List[str] = []
                for bullet in bullets:
                    if not isinstance(bullet, str) or not bullet.strip():
                        continue
                    if self._is_language_compliant(bullet, language):
                        rewritten_bullets.append(bullet)
                    else:
                        rewritten_bullets.append(await self._rewrite_text_language(bullet, language))
                slide["bullets"] = rewritten_bullets
            
            subnarrations = slide.get("subnarrations") or []
            if isinstance(subnarrations, list):
                for sub in subnarrations:
                    if not isinstance(sub, dict):
                        continue
                    summary = sub.get("summary")
                    if isinstance(summary, str) and summary.strip() and not self._is_language_compliant(summary, language):
                        sub["summary"] = await self._rewrite_text_language(summary, language)

    def _is_language_compliant(self, text: str, language: str) -> bool:
        """Heuristic check to detect unwanted script mixing."""
        if not text.strip():
            return True
        
        devanagari = re.compile(r"[\u0900-\u097F]")
        gujarati = re.compile(r"[\u0A80-\u0AFF]")
        has_devanagari = bool(devanagari.search(text))
        has_gujarati = bool(gujarati.search(text))
        
        if language == "Hindi":
            if has_gujarati:
                return False
            return has_devanagari or not text.strip()
        
        if language == "Gujarati":
            if has_devanagari:
                return False
            return has_gujarati or not text.strip()
        
        if language == "English":
            # Allow English + numerals but reject Indic scripts
            return not (has_devanagari or has_gujarati)
        
        return True

    async def _rewrite_text_language(self, text: str, language: str) -> str:
        """Use the LLM to translate/match content to the requested language."""
        normalized = text.strip()
        if not normalized or not self._client:
            return text
        
        cache_key = (normalized, language)
        if cache_key in self._rewrite_cache:
            return self._rewrite_cache[cache_key]
        
        system_prompt = f"You are a precise translator who writes perfectly in {language}."
        user_prompt = (
            f"Convert the following content into {language}. "
            f"Preserve meaning, formatting, and approximate length. "
            f"Return ONLY the converted text without explanations.\n\n"
            f"CONTENT:\n{normalized}"
        )
        
        try:
            completion = await asyncio.to_thread(
                self._client.chat.completions.create,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                model="llama-3.1-8b-instant",
                temperature=0.0,
                max_tokens=min(2048, max(400, len(normalized) * 2)),
            )
            rewritten = completion.choices[0].message.content.strip()
            if rewritten:
                self._rewrite_cache[cache_key] = rewritten
                return rewritten
        except Exception as exc:
            print(f"⚠️ Language rewrite failed: {exc}")
        
        return text
    
    def _enforce_minimum_narration(
        self,
        slides: List[Dict[str, Any]],
        language: str,
        *,
        requested_duration: Optional[int] = None,
    ) -> None:
        """Pad narrations to satisfy minimum word requirements."""
        for idx, slide in enumerate(slides):
            number = slide.get("number") or idx + 1
            min_words = self._get_minimum_word_requirement(
                number,
                requested_duration=requested_duration,
            )
            if not min_words:
                continue
            
            narration = str(slide.get("narration", "")).strip()
            current_words = len(narration.split()) if narration else 0
            if current_words >= min_words:
                continue
            
            padding = self._build_padding_text(
                slide=slide,
                language=language,
                additional_words=min_words - current_words,
            )
            slide["narration"] = f"{narration}\n\n{padding}".strip() if narration else padding
    
    def _get_minimum_word_requirement(
        self,
        slide_number: int,
        *,
        requested_duration: Optional[int] = None,
    ) -> int:
        """Get minimum word count for each slide."""
        if not requested_duration:
            if slide_number == 1:
                return 100
            if 4 <= slide_number <= 7:
                return 250
            if slide_number == 8:
                return 180
            return 0

        word_targets = _get_word_targets(requested_duration)

        def _avg_from_range(value: str) -> int:
            numbers = [int(match) for match in re.findall(r"\d+", value or "")]
            if not numbers:
                return 0
            if len(numbers) == 1:
                return numbers[0]
            return sum(numbers[:2]) // 2

        intro_target = _avg_from_range(word_targets.get("intro", ""))
        normal_target = _avg_from_range(word_targets.get("normal", ""))
        deep_target = _avg_from_range(word_targets.get("deep", ""))

        intro_min = max(150, int(round(intro_target * 0.9))) if intro_target else 200
        normal_min = max(220, int(round(normal_target * 0.9))) if normal_target else 280
        deep_min = max(320, int(round(deep_target * 0.85))) if deep_target else 360
        quiz_min = max(180, int(round(normal_min * 0.6)))
        if slide_number == 1:
            return intro_min
        if slide_number in (2, 3, 8):
            return normal_min
        if 4 <= slide_number <= 7:
            return deep_min
        if slide_number == 9:
            return quiz_min
        return 0
    
    def _build_padding_text(
        self,
        *,
        slide: Dict[str, Any],
        language: str,
        additional_words: int,
    ) -> str:
        """Build padding text for insufficient narration."""
        title = str(slide.get("title") or f"Slide {slide.get('number', '')}").strip()
        bullets = slide.get("bullets") or []
        bullet_summary = ", ".join(bullets[:3]) if bullets else ""
        
        if language == "Hindi":
            intro = f"{title} विषय को समझाते समय हम उदाहरणों से अवधारणा को गहराई से स्पष्ट करते हैं."
            if bullet_summary:
                intro += f" मुख्य बिंदु: {bullet_summary}."
        elif language == "Gujarati":
            intro = f"{title} વિષયને વિદ્યાર્થી સમજી શકે તે માટે અમે સાદા ઉદાહરણોથી દરેક પગલું સમજાવીએ છીએ."
            if bullet_summary:
                intro += f" મુખ્ય મુદ્દા: {bullet_summary}."
        else:
            intro = f"While covering {title}, we guide students through each idea with relatable stories."
            if bullet_summary:
                intro += f" Key ideas: {bullet_summary}."
        
        sentences = [intro]
        filler_bank = self._get_language_fillers(language)
        filler_index = 0
        
        while len(" ".join(sentences).split()) < additional_words:
            sentences.append(filler_bank[filler_index % len(filler_bank)])
            filler_index += 1
        
        return " ".join(sentences)
    
    @staticmethod
    def _get_language_fillers(language: str) -> List[str]:
        """Get language-specific filler sentences."""
        hindi_fillers = [
            "हम सरल भाषा में कारण और परिणाम बताते हैं ताकि हर छात्र अवधारणा को अपनी दैनिक जिंदगी से जोड़ सके.",
            "अभ्यास प्रश्नों के माध्यम से हम बच्चों से त्वरित प्रतिक्रिया लेते हैं और उनकी शंकाओं को वहीं दूर करते हैं.",
            "हर चरण के बाद संक्षिप्त पुनरावृत्ति कराते हैं ताकि मुख्य विचार लंबे समय तक याद रहें.",
            "वास्तविक जीवन के उदाहरणों से सिद्धांत को जोड़कर हम सीखने को रोचक और भरोसेमंद बनाते हैं.",
            "समूह गतिविधियों से सहयोगी अधिगम को बढ़ावा देते हैं और विद्यार्थियों को चर्चा के लिए प्रेरित करते हैं.",
        ]
        
        gujarati_fillers = [
            "અમે સરળ શબ્દોમાં કારણ અને પરિણામ સમજાવીએ છીએ જેથી દરેક વિદ્યાર્થી વિચારને પોતાના અનુભવ સાથે જોડે.",
            "નાના પ્રશ્નો દ્વારા તાત્કાલિક પ્રતિસાદ લઈ ભૂલને તરત સુધારીએ છીએ.",
            "દરેક વિભાગ પછી ઝડપી પુનરાવર્તન કરીએ છીએ જેથી મુખ્ય વિચારો મનમાં મજબૂત બને.",
            "દરરોજના જીવનના ઉદાહરણો દ્વારા સિદ્ધાંતને જીવંત બનાવીને શીખવાનું આનંદમય કરીએ છીએ.",
            "સમૂહ ચર્ચાઓથી સહકાર વધે છે અને વિદ્યાર્થીઓ આત્મવિશ્વાસથી પોતાના વિચારો રજૂ કરે છે.",
        ]
        
        english_fillers = [
            "We pause after each sub-topic to let students reflect and share their understanding.",
            "Short reflective questions strengthen recall and give quick glimpses of progress.",
            "Linking concepts with everyday decisions keeps the narrative practical and memorable.",
            "Hands-on mini activities ensure that theory converts into skill and confidence.",
            "We end blocks with concise recaps so learners feel ready for next challenges.",
        ]
        
        if language == "Hindi":
            return hindi_fillers
        if language == "Gujarati":
            return gujarati_fillers
        return english_fillers

    def _attach_subtopics_to_slides(self, slides: List[Dict[str, Any]]) -> None:
        """Add subnarrations for slides 4-7 directly from their narration.
        NO subtopics field - only subnarrations."""
        if not slides:
            return

        for idx, slide in enumerate(slides):
            number = slide.get("number") or (idx + 1)
            try:
                slide_number = int(number)
            except (TypeError, ValueError):
                slide_number = idx + 1

            # Remove any old subtopics field from ALL slides
            slide.pop("subtopics", None)

            # Add subnarrations ONLY for slides 4-7
            if 4 <= slide_number <= 7:
                narration = str(slide.get("narration", "")).strip()
                slide_title = str(slide.get("title", "")).strip()
                slide["subnarrations"] = self._extract_subnarrations_from_narration(
                    narration=narration,
                    slide_title=slide_title
                )
            else:
                # Ensure no subnarrations for other slides
                slide.pop("subnarrations", None)

    
    def _extract_subnarrations_from_narration(self, narration: str, slide_title: str) -> List[Dict[str, str]]:
        """Extract ONE subnarration from narration (40-60 word summary).
        Returns a list with exactly ONE subnarration object."""
        
        normalized = (narration or "").strip()
        if not normalized:
            return []

        # Split into sentences
        sentences = re.split(r'(?<=[.!?।])\s+', normalized)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        if not sentences:
            return []

        # Take first 2-3 sentences for summary (approximately 40-60 words)
        summary_sentences = sentences[:3]  # Take first 3 sentences
        summary_text = " ".join(summary_sentences)
        
        # Ensure it's within 40-60 words
        words = summary_text.split()
        if len(words) > 60:
            # Trim to approximately 50-60 words
            summary_text = " ".join(words[:55])
            # Try to end at sentence boundary
            last_period = summary_text.rfind('.')
            if last_period > 30:
                summary_text = summary_text[:last_period + 1]
        
        # Return ONLY ONE subnarration
        return [{
            "title": slide_title,
            "summary": summary_text.strip()
        }]

    @staticmethod
    def _chunk_text(text: str, *, chunk_size: int = 120) -> List[str]:
        """Chunk narration into word-based segments."""
        words = text.split()
        if not words:
            return []

        chunks: List[str] = []
        for start in range(0, len(words), chunk_size):
            chunk_words = words[start : start + chunk_size]
            chunk_text = " ".join(chunk_words).strip()
            if len(chunk_words) < 25 and chunks:
                chunks[-1] = (chunks[-1] + " " + chunk_text).strip()
            else:
                chunks.append(chunk_text)

        return [chunk for chunk in chunks if chunk]

    @staticmethod
    def _build_subtopic_title(block: str, index: int) -> str:
        """Create a concise subtopic title from a narration block."""
        if not block:
            return f"Key Idea {index + 1}"

        sentences = re.split(r"(?<=[.!?])\s+", block.strip())
        candidate = sentences[0].strip() if sentences else block.strip()
        words = candidate.split()

        if len(words) > 12:
            candidate = " ".join(words[:12]).rstrip(",;:-") + "..."

        return candidate or f"Key Idea {index + 1}"