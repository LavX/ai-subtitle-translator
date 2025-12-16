#!/usr/bin/env python3
"""
BATTLE ROYALE: 50 Lines x 5 Rounds - Speed Competition for Hungarian Translation
Winner Selection: First to reach threshold with speed tiebreaker
"""
import json
import subprocess
import time
import concurrent.futures
from datetime import datetime
import statistics

# API Configuration
API_KEY = "INSERT_YOUR_API"

# Battle Royale Configuration
TOTAL_LINES = 50
THRESHOLD_HUNGARIAN_PERCENT = 80  # 80% = 40 out of 50 lines
TOTAL_ROUNDS = 5
MAX_ROUND_TIME = 60  # 1 minute per round
HUNGARIAN_THRESHOLD = int(TOTAL_LINES * THRESHOLD_HUNGARIAN_PERCENT / 100)

# Generate 50 test lines
TEST_LINES = [
    {"position": i, "line": f"This is test line number {i+1} that needs to be translated to Hungarian."}
    for i in range(TOTAL_LINES)
]

# Additional variety
VARIED_LINES = [
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
    "The traffic was terrible this morning on my way here.",
    "She plays the piano beautifully at concerts.",
    "They are planning a surprise party for her birthday.",
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
    "The garden flowers are blooming beautifully this season.",
    "Can you recommend a good restaurant nearby?",
    "The presentation went better than expected today.",
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
    "He speaks three languages fluently including English.",
    "The hotel room has a fantastic ocean view.",
    "We celebrated our anniversary at a fancy restaurant.",
    "The dog is sleeping peacefully on the couch.",
    "I need to call my parents later tonight.",
    "The exam results will be announced tomorrow morning.",
    "She always brings homemade cookies to the office.",
    "The winter snow makes everything look magical.",
]

# Use varied lines for the test
TEST_LINES = [{"position": i, "line": VARIED_LINES[i % len(VARIED_LINES)]} for i in range(TOTAL_LINES)]

# BATTLE ROYALE CONTESTANTS
CONTESTANTS = [
    ("anthropic/claude-haiku-4.5", "Claude Haiku 4.5", "Premium"),
    ("moonshotai/kimi-k2-0905:exacto", "Kimi K2 Exacto", "Premium"),
    ("allenai/olmo-3-32b-think:free", "Allen AI Olmo (FREE)", "Free"),
    ("tngtech/deepseek-r1t-chimera:free", "TNG DeepSeek (FREE)", "Free"),
    ("amazon/nova-2-lite-v1:free", "Amazon Nova (FREE)", "Free")
]

def test_model_round(model_id, model_name, category, round_num):
    """Test single model in one round"""
    round_start_time = time.time()
    
    try:
        # Submit job
        result = subprocess.run([
            'curl', '-s', '-X', 'POST', "http://localhost:8765/api/v1/jobs/translate/content",
            '-H', 'Content-Type: application/json',
            '-d', json.dumps({
                "sourceLanguage": "en",
                "targetLanguage": "hu",
                "lines": TEST_LINES,
                "config": {
                    "apiKey": API_KEY,
                    "model": model_id,
                    "temperature": 0.3
                }
            })
        ], capture_output=True, text=True, check=True, timeout=10)
        
        submit_time = time.time() - round_start_time
        job_data = json.loads(result.stdout)
        job_id = job_data['jobId']
        
        # Monitor job
        while (time.time() - round_start_time) < MAX_ROUND_TIME:
            try:
                result = subprocess.run([
                    'curl', '-s', f"http://localhost:8765/api/v1/jobs/{job_id}"
                ], capture_output=True, text=True, check=True, timeout=5)
                
                job_data = json.loads(result.stdout)
                status = job_data.get('status', 'unknown')
                
                if status == 'completed':
                    total_time = time.time() - round_start_time
                    
                    if job_data.get('result', {}).get('lines'):
                        lines = job_data['result']['lines']
                        hungarian_count = sum(
                            1 for line in lines 
                            if any(char in line['line'] for char in 'áéíóúöüőűÁÉÍÓÚÖÜŐŰ')
                        )
                        total_lines = len(lines)
                        success_rate = (hungarian_count / total_lines) * 100
                        
                        return {
                            'name': model_name,
                            'id': model_id,
                            'category': category,
                            'round': round_num,
                            'hungarian_count': hungarian_count,
                            'total_lines': total_lines,
                            'success_rate': success_rate,
                            'total_time': total_time,
                            'status': 'victory' if hungarian_count >= HUNGARIAN_THRESHOLD else 'partial',
                            'tokens': job_data.get('result', {}).get('tokens_used', 0)
                        }
                        
                elif status == 'failed':
                    return {
                        'name': model_name,
                        'id': model_id,
                        'category': category,
                        'round': round_num,
                        'status': 'failed',
                        'total_time': time.time() - round_start_time
                    }
                    
                time.sleep(0.5)
                
            except Exception:
                time.sleep(1)
                
        return {
            'name': model_name,
            'id': model_id,
            'category': category,
            'round': round_num,
            'status': 'timeout',
            'total_time': MAX_ROUND_TIME
        }
        
    except Exception as e:
        return {
            'name': model_name,
            'id': model_id,
            'category': category,
            'round': round_num,
            'status': 'error',
            'error': str(e)[:50]
        }

def battle_round(round_num):
    """Execute one battle round"""
    print(f"\n{'='*80}")
    print(f"⚔️  ROUND {round_num} - 50 Lines - {HUNGARIAN_THRESHOLD}+ Hungarian to WIN!")
    print(f"{'='*80}")
    
    round_results = []
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        futures = {
            executor.submit(test_model_round, m[0], m[1], m[2], round_num): m
            for m in CONTESTANTS
        }
        
        print("🌪️  All contestants racing...\n")
        
        for future in concurrent.futures.as_completed(futures, timeout=MAX_ROUND_TIME + 30):
            try:
                result = future.result(timeout=5)
                round_results.append(result)
                
                if result.get('status') == 'victory':
                    print(f"✅ {result['name']}: {result['hungarian_count']}/{result['total_lines']} HU ({result['success_rate']:.0f}%) in {result['total_time']:.1f}s")
                elif result.get('status') == 'partial':
                    print(f"⚠️  {result['name']}: {result['hungarian_count']}/{result['total_lines']} HU ({result['success_rate']:.0f}%) in {result['total_time']:.1f}s")
                else:
                    print(f"❌ {result['name']}: {result['status']}")
            except:
                pass
    
    # Find round winner
    winners = [r for r in round_results if r.get('status') == 'victory']
    if winners:
        fastest = min(winners, key=lambda x: x['total_time'])
        print(f"\n🏆 ROUND {round_num} WINNER: {fastest['name']} - {fastest['total_time']:.1f}s - {fastest['success_rate']:.0f}%")
        return fastest
    
    # No victory, pick best performer
    best = max(round_results, key=lambda x: x.get('hungarian_count', 0), default=None)
    if best:
        print(f"\n🥈 ROUND {round_num} BEST: {best['name']} - {best.get('hungarian_count', 0)} Hungarian")
    return best

def main():
    """Battle Royale Championship"""
    print("🔥 BATTLE ROYALE: 50 Lines - Hungarian Translation Championship 🔥")
    print(f"Threshold: {HUNGARIAN_THRESHOLD}/{TOTAL_LINES} ({THRESHOLD_HUNGARIAN_PERCENT}%) Hungarian")
    print(f"Time Limit: {MAX_ROUND_TIME}s per round")
    print(f"Rounds: {TOTAL_ROUNDS}")
    print("-" * 80)
    
    round_winners = []
    
    for round_num in range(1, TOTAL_ROUNDS + 1):
        winner = battle_round(round_num)
        if winner:
            round_winners.append(winner)
    
    # Final standings
    print("\n" + "="*80)
    print("🏆 FINAL CHAMPIONSHIP STANDINGS 🏆")
    print("="*80)
    
    # Count wins per model
    win_counts = {}
    speed_totals = {}
    
    for w in round_winners:
        model_id = w['id']
        if model_id not in win_counts:
            win_counts[model_id] = {'wins': 0, 'name': w['name'], 'category': w['category'], 'times': []}
        if w.get('status') == 'victory':
            win_counts[model_id]['wins'] += 1
        win_counts[model_id]['times'].append(w.get('total_time', MAX_ROUND_TIME))
    
    # Sort by wins, then by average speed
    standings = sorted(
        win_counts.items(),
        key=lambda x: (x[1]['wins'], -statistics.mean(x[1]['times']) if x[1]['times'] else 0),
        reverse=True
    )
    
    for rank, (model_id, stats) in enumerate(standings, 1):
        avg_time = statistics.mean(stats['times']) if stats['times'] else 0
        print(f"{rank}. {stats['name']} ({stats['category']})")
        print(f"   Wins: {stats['wins']}/{TOTAL_ROUNDS} | Avg Time: {avg_time:.1f}s")
    
    if standings:
        champion = standings[0]
        print(f"\n🎉 CHAMPION: {champion[1]['name']} ({champion[1]['category']}) 🎉")
        print(f"   Wins: {champion[1]['wins']}/{TOTAL_ROUNDS}")

if __name__ == "__main__":
    main()