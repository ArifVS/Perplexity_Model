import math
ngram_model = {
    ('I',): {'like': 0.5, 'love': 0.5},
    ('like',): {'apples': 1.0},
    ('love',): {'apples': 1.0},
}
test_sentence = ['I', 'like', 'apples']
def calculate_perplexity(model, sentence, n=2):
    N = len(sentence) - (n - 1)
    log_prob_sum = 0.0
    print("Predictions:")
    for i in range(N):
        context = tuple(sentence[i:i+n-1])
        word = sentence[i+n-1]
        prob = model.get(context, {}).get(word, 1e-6)
        print(f"Context: {context} -> Next word: '{word}' | Predicted Prob: {prob}")
        log_prob_sum += math.log(prob)
    perplexity = math.exp(-log_prob_sum / N)
    return perplexity

perplexity = calculate_perplexity(ngram_model, test_sentence, n=2)
print("Perplexity:", perplexity)
