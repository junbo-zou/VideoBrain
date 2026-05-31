import re
import asyncio
import random
import os
import json
from openai import AsyncOpenAI, BadRequestError

default_api_key = ""
default_api_base = ""

openai_api_base_list = [
    os.environ.get("LLM_AS_A_JUDGE_BASE", default_api_base),
]

async_client_list = []
for api_base in openai_api_base_list:
    api_key = os.environ.get("DASHSCOPE_API_KEY") or os.environ.get("OPENAI_API_KEY") or default_api_key
    client = AsyncOpenAI(
        api_key=api_key,
        base_url=api_base,
        timeout=30.0,
    )
    async_client_list.append(client)

default_model = os.environ.get("LLM_AS_A_JUDGE_MODEL", "deepseek-v3.2-exp")
model_name_list = [default_model] * len(async_client_list)

# MC: strict 0/1 scoring
System_PROMPT_MC_JSON = """You are an AI judge for multiple-choice video questions.

Your task:
1. Compare the "model response" with the "ground truth" answer (a single letter like A, B, C, D)
2. Assign score: 1.0 if correct, 0.0 if wrong

Scoring rules (STRICT):
- Correct choice: 1.0
- Wrong choice: 0.0
- Multiple choices (e.g., "A and B"): 0.0
- Missing choice: 0.0
- Any deviation from ground truth: 0.0

Output format - Return ONLY a valid JSON:
{"score": 1.0}  or  {"score": 0.0}

IMPORTANT: Only two possible scores: 1.0 or 0.0, nothing in between."""

USER_PROMPT_MC_JSON = """Question: {Question}
Ground Truth: {Truth}
Model Response: {ModelAnswer}

Output JSON:"""

# OE: semantic similarity scoring
System_PROMPT_OE_JSON = """You are responsible for proofreading the answers, you need to give a score to the model's answer by referring to the standard answer,
based on the given question. The full score is 1 point and the minimum score is 0 points.

Output format - Return ONLY a valid JSON:
{"score": <score>}

The evaluation criteria require that the closer the model's answer is to the standard answer, the higher the score."""

USER_PROMPT_OE_JSON = """Question: {Question}
Ground Truth: {Truth}
Model Response: {ModelAnswer}

Output JSON:"""


def compact_image_pads(text: str) -> str:
    compacted_text = re.sub(r'(<\|image_pad\|>)+', r'<|image_pad|>', text)
    return compacted_text


def print_answer(text: str, ground_truth: str):
    print("predict_str:", compact_image_pads(text))
    print("ground_truth:", ground_truth)


def validate_format_v2(predict_str: str, extra_info=None):
    # Strip format examples from turn prompts before counting tags
    turn_prompt_normal = "Start with <thinking>. Format strictly as: <thinking>...</thinking><tool_call>...</tool_call> or <thinking>...</thinking><answer>...</answer>"
    turn_prompt_final = "Start with <thinking>. Format strictly as: <thinking>...</thinking><answer>...</answer>"

    predict_str = predict_str.replace(turn_prompt_normal, "")
    predict_str = predict_str.replace(turn_prompt_final, "")

    # 1. Tag pairing checks
    thinking_open_count = predict_str.count('<thinking>')
    thinking_close_count = predict_str.count('</thinking>')
    if thinking_open_count != thinking_close_count:
        print(f"\n{'='*80}")
        print(f"[FORMAT ERROR] thinking tag mismatch: {thinking_open_count} open vs {thinking_close_count} close")
        print(f"{'='*80}\n")
        return False, None

    answer_open_count = predict_str.count('<answer>')
    answer_close_count = predict_str.count('</answer>')
    if answer_open_count != answer_close_count:
        print(f"\n{'='*80}")
        print(f"[FORMAT ERROR] answer tag mismatch: {answer_open_count} open vs {answer_close_count} close")
        print(f"{'='*80}\n")
        return False, None

    tool_call_open_count = predict_str.count('<tool_call>')
    tool_call_close_count = predict_str.count('</tool_call>')
    if tool_call_open_count != tool_call_close_count:
        print(f"\n{'='*80}")
        print(f"[FORMAT ERROR] tool_call tag mismatch: {tool_call_open_count} open vs {tool_call_close_count} close")
        print(f"{'='*80}\n")
        return False, None

    # thinking count must equal tool_call count + answer count
    if thinking_open_count != tool_call_open_count + answer_open_count:
        print(f"\n{'='*80}")
        print(f"[FORMAT ERROR] thinking count ({thinking_open_count}) != tool_call ({tool_call_open_count}) + answer ({answer_open_count})")
        print(f"{'='*80}\n")
        return False, None

    # 2. Non-empty content checks
    thinking_pattern = r'<thinking>(.*?)</thinking>'
    thinking_matches = re.findall(thinking_pattern, predict_str, re.DOTALL)
    for idx, thinking_content in enumerate(thinking_matches):
        if not thinking_content.strip():
            print(f"\n{'='*80}")
            print(f"[FORMAT ERROR] Empty thinking tag found (index {idx})")
            print(f"{'='*80}\n")
            return False, None

    answer_pattern = r'<answer>(.*?)</answer>'
    answer_matches = re.findall(answer_pattern, predict_str, re.DOTALL)
    if len(answer_matches) == 0:
        print(f"\n{'='*80}")
        print(f"[FORMAT ERROR] No answer tag found")
        print(f"{'='*80}\n")
        return False, None

    model_answer = answer_matches[-1].strip()
    if not model_answer:
        print(f"\n{'='*80}")
        print(f"[FORMAT ERROR] Empty answer content")
        print(f"{'='*80}\n")
        return False, None

    tool_call_pattern = r'<tool_call>(.*?)</tool_call>'
    tool_call_matches = re.findall(tool_call_pattern, predict_str, re.DOTALL)
    for idx, tool_call_content in enumerate(tool_call_matches):
        if not tool_call_content.strip():
            print(f"\n{'='*80}")
            print(f"[FORMAT ERROR] Empty tool_call tag found (index {idx})")
            print(f"{'='*80}\n")
            return False, None

    # 3. Tool_call JSON validation
    action_num = 0
    uniform_sample_history = []
    clip_sample_prompt_history = []

    for tool_call_content in tool_call_matches:
        try:
            tool_data = json.loads(tool_call_content.strip())
            tool_name = tool_data.get("name")

            if tool_name == "uniform_sample":
                arguments = tool_data.get("arguments", {})
                start_frame = arguments.get("start_frame")
                end_frame = arguments.get("end_frame")

                if start_frame is None or end_frame is None:
                    print(f"\n{'='*80}")
                    print(f"[FORMAT ERROR] uniform_sample missing start_frame or end_frame")
                    print(f"{'='*80}\n")
                    return False, None

                try:
                    start_frame = int(start_frame)
                    end_frame = int(end_frame)
                except (ValueError, TypeError) as e:
                    print(f"\n{'='*80}")
                    print(f"[FORMAT ERROR] Invalid parameter type: {e}")
                    print(f"{'='*80}\n")
                    return False, None

                if start_frame >= end_frame:
                    print(f"\n{'='*80}")
                    print(f"[FORMAT ERROR] start_frame({start_frame}) >= end_frame({end_frame})")
                    print(f"{'='*80}\n")
                    return False, None

                # Reject duplicate or near-duplicate calls (within ±1 frame)
                for prev_start, prev_end in uniform_sample_history:
                    is_exact_match = (start_frame == prev_start and end_frame == prev_end)
                    is_approx_match = (abs(start_frame - prev_start) <= 1 and abs(end_frame - prev_end) <= 1)
                    if is_exact_match or is_approx_match:
                        print(f"\n{'='*80}")
                        print(f"[FORMAT ERROR] Duplicate uniform_sample detected: ({start_frame}, {end_frame}) vs previous ({prev_start}, {prev_end})")
                        print(f"{'='*80}\n")
                        return False, None

                uniform_sample_history.append((start_frame, end_frame))
                action_num += 1

            elif tool_name == "clip_sample":
                arguments = tool_data.get("arguments", {})
                start_frame = arguments.get("start_frame")
                end_frame = arguments.get("end_frame")
                prompt = arguments.get("prompt")

                if start_frame is None or end_frame is None or prompt is None:
                    print(f"\n{'='*80}")
                    print(f"[FORMAT ERROR] clip_sample missing required parameters")
                    print(f"{'='*80}\n")
                    return False, None

                try:
                    start_frame = int(start_frame)
                    end_frame = int(end_frame)
                except (ValueError, TypeError) as e:
                    print(f"\n{'='*80}")
                    print(f"[FORMAT ERROR] Invalid parameter type: {e}")
                    print(f"{'='*80}\n")
                    return False, None

                if start_frame >= end_frame:
                    print(f"\n{'='*80}")
                    print(f"[FORMAT ERROR] start_frame({start_frame}) >= end_frame({end_frame})")
                    print(f"{'='*80}\n")
                    return False, None

                if not isinstance(prompt, str) or not prompt.strip():
                    print(f"\n{'='*80}")
                    print(f"[FORMAT ERROR] clip_sample prompt is empty or invalid")
                    print(f"{'='*80}\n")
                    return False, None

                prompt_normalized = prompt.strip()
                if prompt_normalized in clip_sample_prompt_history:
                    print(f"\n{'='*80}")
                    print(f"[FORMAT ERROR] Duplicate clip_sample prompt detected: '{prompt_normalized}'")
                    print(f"{'='*80}\n")
                    return False, None

                clip_sample_prompt_history.append(prompt_normalized)
                action_num += 1

            else:
                print(f"\n{'='*80}")
                print(f"[FORMAT ERROR] Unknown tool name: {tool_name}")
                print(f"{'='*80}\n")
                return False, None

        except json.JSONDecodeError as e:
            print(f"\n{'='*80}")
            print(f"[FORMAT ERROR] JSON parsing failed: {e}")
            print(f"{'='*80}\n")
            return False, None

    return True, {
        'model_answer': model_answer,
        'action_num': action_num
    }


def parse_json_score(response: str) -> float:
    # Method 1: direct JSON parse
    try:
        data = json.loads(response)
        score = float(data['score'])
        return max(0.0, min(1.0, score))
    except:
        pass

    # Method 2: regex extract JSON (handles markdown-wrapped responses)
    json_match = re.search(r'\{[^}]*"score"[^}]*:[^}]*\}', response, re.DOTALL)
    if json_match:
        try:
            data = json.loads(json_match.group(0))
            score = float(data['score'])
            return max(0.0, min(1.0, score))
        except:
            pass

    # Method 3: loosest fallback — extract score field value
    score_match = re.search(r'"score"[:\s]+(\d+\.?\d*)', response)
    if score_match:
        score = float(score_match.group(1))
        return max(0.0, min(1.0, score))

    return None


async def score_with_llm_judge(question: str, ground_truth: str, model_answer: str,
                                model_name: str, question_type: str) -> float:
    if question_type == 'mc':
        system_prompt = System_PROMPT_MC_JSON
        user_prompt = USER_PROMPT_MC_JSON.format(
            Question=question,
            Truth=ground_truth,
            ModelAnswer=model_answer
        )
        temperature = 0.0
    else:  # oe
        system_prompt = System_PROMPT_OE_JSON
        user_prompt = USER_PROMPT_OE_JSON.format(
            Question=question,
            Truth=ground_truth,
            ModelAnswer=model_answer
        )
        temperature = 0.0

    client_idx = random.randint(0, len(async_client_list) - 1)
    client = async_client_list[client_idx]

    for attempt in range(8):
        try:
            chat_response = await client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                seed=random.randint(0, 1000000),
                temperature=temperature,
            )
            response = chat_response.choices[0].message.content.strip()

            score = parse_json_score(response)
            if score is not None:
                if question_type == 'mc':
                    score = 1.0 if score >= 0.5 else 0.0
                return score

            print(f'[WARNING] Failed to parse score from response (attempt {attempt+1}/8): {response[:200]}')
            continue

        except BadRequestError as e:
            print(f'[WARNING] BadRequestError (400), returning 0.0: {e}')
            return 0.0
        except Exception as e:
            print(f'[WARNING] LLM judge attempt {attempt+1}/8 failed: {e}')
            continue

    print(f'[ERROR] All 8 attempts failed for {question_type.upper()} question: {question[:50]}...')
    return 0.0


def get_behavior_bonus(classification: str, used_agent: bool) -> float:
    """
    Behavior bonus table (0 or 0.5), called only when acc_score == 1.0:
    | classification | no agent | used agent |
    |----------------|----------|------------|
    | a, b           | 0.5      | 0          |
    | e              | 0.5      | 0.5        |
    | f, g, h        | 0        | 0.5        |
    """
    classification = classification.lower().strip() if classification else ''

    if classification in ['a', 'b']:
        return 0.0 if used_agent else 0.5
    elif classification == 'e':
        return 0.5
    elif classification in ['f', 'g', 'h']:
        return 0.5 if used_agent else 0.0
    else:
        print(f"[WARNING] Unknown classification '{classification}', not in [a, b, e, f, g, h], behavior_bonus = 0")
        return 0.0


async def compute_score_v2(predict_str: str, ground_truth: str, extra_info=None):
    """
    total_score = format_score + acc_score + behavior_bonus
    - format_score: 0 or 0.05
    - acc_score: 0 or 1.0
    - behavior_bonus: 0 or 0.5 (only when acc_score == 1.0)
    """
    # 1. Format check
    is_valid, parsed_data = validate_format_v2(predict_str, extra_info)

    if not is_valid:
        print("\n" + "="*80)
        print("[REWARD] FORMAT ERROR - Total Score: 0")
        print("="*80)
        print(f"[MODEL OUTPUT - FULL]")
        print("-"*40)
        print(compact_image_pads(predict_str))
        print("-"*40)
        print("="*80 + "\n")
        return 0.0, 0.0, 0.0, 0.0

    format_score = 0.05
    model_answer = parsed_data['model_answer']
    action_num = parsed_data['action_num']
    used_agent = action_num >= 1

    # 2. LLM judge
    question_type = extra_info.get('question_type', 'mc') if extra_info else 'mc'
    question = extra_info.get('question', '') if extra_info else ''
    classification = extra_info.get('classification', '') if extra_info else ''

    if question_type == 'mc':
        llm_score = await score_with_llm_judge(
            question=question,
            ground_truth=ground_truth,
            model_answer=model_answer,
            model_name="qwen-flash",
            question_type="mc"
        )
    else:  # oe
        llm_score = await score_with_llm_judge(
            question=question,
            ground_truth=ground_truth,
            model_answer=model_answer,
            model_name="deepseek-v3.2",
            question_type="oe"
        )

    if question_type == 'mc':
        acc_score = 1.0 if llm_score >= 1.0 else 0.0
    else:  # oe
        acc_score = llm_score

    # 3. Behavior bonus
    if acc_score >= 1.0:
        behavior_bonus = get_behavior_bonus(classification, used_agent)
    else:
        # Partial credit for hard questions (f/g/h) where agent was used
        classification_lower = classification.lower().strip() if classification else ''
        if classification_lower in ['f', 'g', 'h'] and used_agent:
            behavior_bonus = 0.2
        else:
            behavior_bonus = 0.0

    # 4. Total
    total_score = format_score + acc_score + behavior_bonus

    # 5. Print report
    print("\n" + "="*80)
    print("[REWARD] EVALUATION REPORT")
    print("="*80)
    print(f"[MODEL OUTPUT - FULL]")
    print("-"*40)
    print(compact_image_pads(predict_str))
    print("-"*40)
    print(f"[PARSED INFO]")
    print(f"  Model Answer: {model_answer}")
    print(f"  Ground Truth: {ground_truth}")
    print(f"  Classification: {classification}")
    print(f"  Used Agent: {'Yes' if used_agent else 'No'} (tool_calls={action_num})")
    print(f"[SCORING]")
    print(f"  Format Score: {format_score:.2f}")
    print(f"  LLM Judge Score: {llm_score:.2f}")
    print(f"  Acc Score: {acc_score:.2f}")
    print(f"  Behavior Bonus: {behavior_bonus:.2f}")
    print(f"  Total Score: {total_score:.2f}")
    print("="*80 + "\n")

    return total_score, format_score, acc_score, behavior_bonus


def compute_score(predict_str: str, ground_truth: str, extra_info=None):
    """
    Synchronous wrapper for compute_score_v2.

    Returns:
        (total_score, format_score, acc_score, behavior_bonus)
    """
    total_score, format_score, acc_score, behavior_bonus = asyncio.run(
        compute_score_v2(predict_str, ground_truth, extra_info)
    )
    return total_score, format_score, acc_score, behavior_bonus
