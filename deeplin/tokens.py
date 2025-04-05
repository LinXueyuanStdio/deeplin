def token_counts(texts: list[str], tokenizer, **kwargs) -> list[int]:
  # 批量分词
  encoded_inputs = tokenizer(texts, **kwargs)

  # 统计每个文本的 token 数
  token_counts = [len(input_ids) for input_ids in encoded_inputs['input_ids']]

  return token_counts

