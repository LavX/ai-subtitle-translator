#!/usr/bin/env python3
"""
ELIMINATION BATTLE ROYALE: Hungarian Translation Championship
Rounds: 5→10→20→30→40→50 lines - Losers get ELIMINATED!
Winner: Last model standing
"""
import json
import subprocess
import time
import concurrent.futures
from datetime import datetime
import statistics

# API Configuration  
API_KEY = ""your_api_key_here""

# Tournament Configuration
ROUNDS = [5, 10, 20, 30, 40, 50]  # Increasing difficulty
MAX_ROUND_TIME = 120  # 2 minutes per round
THRESHOLD_PERCENT = 80  # 80% Hungarian to survive

# Test sentences variety
SENTENCES = [
    "Hello, how are you doing today?",
    "The weather outside is absolutely beautiful.",
    "I am feeling fantastic and thank you for asking.",
    "This is a peaceful moment that I'm enjoying.",
    "Thank you so much for your kind thoughts.",
    "What time is the meeting scheduled for tomorrow?",
    "Can you please help me with this problem?",
    "I really appreciate your assistance with this.",
    "The sunset was incredible yesterday evening.",
    "We should go for a walk in the park later.",
    "Have you seen the latest news about technology?",
    "The restaurant around the corner makes great food.",
    "I need to finish this project by Friday.",
    "My favorite movie is playing at the cinema tonight.",
    "The traffic was terrible this morning.",
    "She plays the piano beautifully at concerts.",
    "They are planning a surprise party for her.",
    "The book you recommended was really interesting.",
    "We should meet for coffee sometime next week.",
    "The children are playing happily in the garden.",
    "I can't believe how fast time flies these days.",
    "This coffee tastes amazing in the morning.",
    "The mountain views from here are breathtaking.",
    "Please let me know if you need anything else.",
    "The train will arrive at the station soon.",
    "Her dress looks absolutely stunning tonight.",
    "We had a wonderful time at the beach yesterday.",
    "The concert tickets sold out within minutes.",
    "I forgot my umbrella at the office again.",
    "The garden flowers are blooming beautifully.",
    "Can you recommend a good restaurant nearby?",
    "The presentation went better than expected.",
    "I need to buy groceries on my way home.",
    "The museum exhibit was fascinating to explore.",
    "She runs every morning before work starts.",
    "The cake you baked was absolutely delicious.",
    "We should plan our vacation for next summer.",
    "The new software update fixed all the bugs.",
    "I love spending time with my family on weekends.",
    "The lecture was very informative and engaging.",
    "Please remember to lock the door when leaving.",
    "The autumn leaves are changing colors beautifully.",
    "He speaks three languages fluently now.",
    "The hotel room has a fantastic ocean view.",
    "We celebrated our anniversary at a restaurant.",
    "The dog is sleeping peacefully on the couch.",
    "I need to call my parents later tonight.",
    "The exam results will be announced tomorrow.",
    "She always brings homemade cookies to work.",
    "The winter snow makes everything look magical.",
]

# ALL CONTESTANTS - Champions + Mid-tier models
ALL_CONTESTANTS = [
    # Premium Champions
    ("anthropic/claude-haiku-4.5", "Claude Haiku 4.5", "Premium"),
    ("moonshotai/kimi-k2-0905:exacto", "Kimi K2 Exacto", "Premium"),
    ("anthropic/claude-sonnet-4.5", "Claude Sonnet 4.5", "Premium"),
    
    # Free Champions
    ("allenai/olmo-3-32b-think:free", "Allen AI Olmo (FREE)", "Free"),
    ("tngtech/deepseek-r1t-chimera:free", "TNG DeepSeek R1T (FREE)", "Free"),
    
    # Free Mid-tier
    ("amazon/nova-2-lite-v1:free", "Amazon Nova (FREE)", "Free"),
    ("nex-agi/deepseek-v3.1-nex-n1:free", "NEX DeepSeek V3.1 (FREE)", "Free"),
    
    # Good Models (66% success in previous tests)
    ("google/gemini-2.5-flash-preview-09-2025", "Gemini Flash", "Good"),
    ("google/gemini-2.5-flash-lite-preview-09-2025", "Gemini Flash Lite", "Good"),
    ("openai/gpt-oss-120b:exacto", "GPT OSS 120B", "Good"),
    ("meta-llama/llama-4-maverick", "Llama 4 Maverick", "Good"),
    
    # Challengers (untested free models)
    ("nvidia/nemotron-3-nano-30b-a3b:free", "NVIDIA Nemotron (FREE)", "Challenger"),
    ("cognitivecomputations/dolphin-mistral-24b-venice-edition:free", "Dolphin Mistral (FREE)", "Challenger"),
]

def generate_test_lines(count):
    """Generate test lines for a round"""
    return [{"position": i, "line": SENTENCES[i % len(SENTENCES)]} for i in range(count)]

def test_model(model_id, model_name, category, lines, round_name):
    """Test a single model"""
    start_time = time.time()
    
    try:
        # Submit job
        result = subprocess.run([
            'curl', '-s', '-X', 'POST', "http://localhost:8765/api/v1/jobs/translate/content",
            '-H', 'Content-Type: application/json',
            '-d', json.dumps({
                "sourceLanguage": "en",
                "targetLanguage": "hu",
                "lines": lines,
                "config": {
                    "apiKey": API_KEY,
                    "model": model_id,
                    "temperature": 0.3
                }
            })
        ], capture_output=True, text=True, check=True, timeout=15)
        
        job_data = json.loads(result.stdout)
        job_id = job_data['jobId']
        
        # Monitor job
        while (time.time() - start_time) < MAX_ROUND_TIME:
            try:
                result = subprocess.run([
                    'curl', '-s', f"http://localhost:8765/api/v1/jobs/{job_id}"
                ], capture_output=True, text=True, check=True, timeout=10)
                
                job_data = json.loads(result.stdout)
                status = job_data.get('status', 'unknown')
                
                if status == 'completed':
                    total_time = time.time() - start_time
                    
                    if job_data.get('result', {}).get('lines'):
                        result_lines = job_data['result']['lines']
                        hungarian_count = sum(
                            1 for line in result_lines 
                            if any(char in line['line'] for char in 'áéíóúöüőűÁÉÍÓÚÖÜŐŰ')
                        )
                        total_lines = len(result_lines)
                        success_rate = (hungarian_count / total_lines) * 100 if total_lines > 0 else 0
                        threshold = int(total_lines * THRESHOLD_PERCENT / 100)
                        
                        return {
                            'name': model_name,
                            'id': model_id,
                            'category': category,
                            'hungarian_count': hungarian_count,
                            'total_lines': total_lines,
                            'success_rate': success_rate,
                            'total_time': total_time,
                            'survived': hungarian_count >= threshold,
                            'tokens': job_data.get('result', {}).get('tokens_used', 0)
                        }
                        
                elif status == 'failed':
                    return {
                        'name': model_name,
                        'id': model_id,
                        'category': category,
                        'survived': False,
                        'total_time': time.time() - start_time,
                        'error': 'Job failed'
                    }
                    
                time.sleep(0.5)
                
            except Exception:
                time.sleep(1)
                
        # Timeout
        return {
            'name': model_name,
            'id': model_id,
            'category': category,
            'survived': False,
            'total_time': MAX_ROUND_TIME,
            'error': 'Timeout'
        }
        
    except Exception as e:
        return {
            'name': model_name,
            'id': model_id,
            'category': category,
            'survived': False,
            'error': str(e)[:50]
        }

def elimination_round(round_num, line_count, survivors):
    """Execute one elimination round"""
    threshold = int(line_count * THRESHOLD_PERCENT / 100)
    
    print(f"\n{'='*90}")
    print(f"⚔️  ROUND {round_num}: {line_count} LINES - {threshold}+ Hungarian to SURVIVE!")
    print(f"   Contestants: {len(survivors)} | Time Limit: {MAX_ROUND_TIME}s")
    print(f"{'='*90}")
    
    lines = generate_test_lines(line_count)
    results = []
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=min(8, len(survivors))) as executor:
        futures = {
            executor.submit(test_model, m[0], m[1], m[2], lines, f"Round {round_num}"): m
            for m in survivors
        }
        
        print(f"🌪️  {len(survivors)} models racing...\n")
        
        for future in concurrent.futures.as_completed(futures, timeout=MAX_ROUND_TIME + 60):
            try:
                result = future.result(timeout=10)
                results.append(result)
                
                if result.get('survived'):
                    emoji = "✅"
                    msg = f"{result['hungarian_count']}/{result['total_lines']} HU ({result['success_rate']:.0f}%) in {result['total_time']:.1f}s"
                elif result.get('error'):
                    emoji = "💀"
                    msg = f"ELIMINATED - {result['error']}"
                else:
                    emoji = "❌"
                    msg = f"ELIMINATED - {result.get('hungarian_count', 0)}/{result.get('total_lines', line_count)} HU ({result.get('success_rate', 0):.0f}%)"
                
                print(f"{emoji} {result['name']} ({result['category']}): {msg}")
                
            except Exception as e:
                print(f"💀 Error: {str(e)[:50]}")
    
    # Determine survivors
    round_survivors = [r for r in results if r.get('survived')]
    eliminated = [r for r in results if not r.get('survived')]
    
    print(f"\n📊 ROUND {round_num} RESULTS:")
    print(f"   Survivors: {len(round_survivors)} | Eliminated: {len(eliminated)}")
    
    if round_survivors:
        # Sort survivors by speed
        round_survivors.sort(key=lambda x: x.get('total_time', 999))
        fastest = round_survivors[0]
        print(f"   🏆 Fastest: {fastest['name']} - {fastest['total_time']:.1f}s")
    
    if eliminated:
        print(f"   💀 Eliminated: {', '.join(e['name'] for e in eliminated)}")
    
    # Return survivor model tuples for next round
    survivor_ids = {r['id'] for r in round_survivors}
    return [(m[0], m[1], m[2]) for m in survivors if m[0] in survivor_ids], results

def main():
    """Run Elimination Battle Royale"""
    print("🔥🔥🔥 ELIMINATION BATTLE ROYALE: Hungarian Translation Championship 🔥🔥🔥")
    print("="*90)
    print(f"Rounds: {' → '.join(str(r) for r in ROUNDS)} lines")
    print(f"Threshold: {THRESHOLD_PERCENT}% Hungarian to survive")
    print(f"Time Limit: {MAX_ROUND_TIME}s per round")
    print(f"Total Contestants: {len(ALL_CONTESTANTS)}")
    print("="*90)
    
    survivors = list(ALL_CONTESTANTS)
    all_results = {}
    
    for round_num, line_count in enumerate(ROUNDS, 1):
        if len(survivors) <= 1:
            break
            
        survivors, results = elimination_round(round_num, line_count, survivors)
        all_results[f"round_{round_num}"] = results
        
        if len(survivors) == 0:
            print("\n💀💀💀 ALL CONTESTANTS ELIMINATED! NO CHAMPION! 💀💀💀")
            break
        elif len(survivors) == 1:
            print(f"\n🎉🏆🎉 CHAMPION DECLARED: {survivors[0][1]} 🎉🏆🎉")
            break
    
    # Final standings
    print("\n" + "="*90)
    print("🏆 FINAL CHAMPIONSHIP STANDINGS 🏆")
    print("="*90)
    
    if survivors:
        # Sort final survivors by their last round performance
        print("\n🥇 CHAMPIONS (Survived all rounds):")
        for i, s in enumerate(survivors, 1):
            print(f"   {i}. {s[1]} ({s[2]})")
    
    # Show elimination order
    print("\n💀 ELIMINATION ORDER:")
    eliminated_order = []
    for round_num in range(1, len(ROUNDS) + 1):
        round_key = f"round_{round_num}"
        if round_key in all_results:
            for r in all_results[round_key]:
                if not r.get('survived') and r['id'] not in [e['id'] for e in eliminated_order]:
                    eliminated_order.append(r)
                    print(f"   Round {round_num}: {r['name']} ({r['category']})")
    
    print("\n" + "="*90)
    print("🏆 BATTLE ROYALE COMPLETE! 🏆")
    print("="*90)

if __name__ == "__main__":
    main()