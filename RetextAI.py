from transformers import T5ForConditionalGeneration, T5Tokenizer
MODEL_NAME = 'cointegrated/rut5-base-paraphraser'
model = T5ForConditionalGeneration.from_pretrained(MODEL_NAME)
tokenizer = T5Tokenizer.from_pretrained(MODEL_NAME)

model.eval();

def paraphrase(text, sequences, beams=15, grams = 4, do_sample=True):
    x = tokenizer(text, return_tensors='pt', padding=True).to(model.device)
    max_size = int(x.input_ids.shape[1] * 1.5 + 10)
    out = model.generate(**x, encoder_no_repeat_ngram_size=grams,
      num_beams=beams, max_length=max_size,
      num_return_sequences=sequences,
      do_sample=do_sample)
    return tokenizer.batch_decode(out, skip_special_tokens=True)



if __name__ == '__main__':

    while True:
      input_text = input('Введите текст для перефразирования: ')
      sequences = int(input('Введите желаемое количество вариантов перефразирования до 15: '))
      paraphrased_text = paraphrase(input_text, sequences)
      print('Перефразированный текст:')

      for element in paraphrased_text:
        print(element)

      result = input('Вас устраивает результат? ' )
      if result == 'Да':
        break
      else:
        print('Повторите')

