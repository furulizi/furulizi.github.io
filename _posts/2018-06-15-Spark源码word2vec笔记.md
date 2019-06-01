---
layout: post
title: "Spark源码word2vec笔记"
date: 2018-06-15 08:09:20 +0300
description: Spark源码word2vec笔记 # Add post description (optional)
img:  # Add image post (optional)
tags: [word2vec]
---

# Spark源码word2vec笔记

> 这里先不详述word2vec原理，重点放在Spark源码的实现。Spark版本实现的是基于Hierarchical Softmax优化的skip-gram模型。


## 1. 建立Huffman树

```
case class VocabWord{
  var word: String, // 词语
  var cn: Int, // 词频
  var point: Array[Int], // 存储路径，即经过的节点
  var code: Array[Int], // huffman编码
  var codeLen: Int // huffman编码长度
}
```
- 建立huffman树分成两步：一是统计词频，按词频降序排序，得到字典中每个词的排序（vocabHash:HashMap[String,Int]，记录词对应的Index）；二是将字典存储成Huffman树；存储Huffman树使用到以上数据结构。
- 注意，以上的Array[VocabWord]和vacabHash是一一对应的。

## 2. 训练过程
> skip-gram 的过程是用滑动窗口去遍历语料段，给定目标词，计算上下文的概率。

### 2.1 文本映射为index

```
val sentences: RDD[Array[Int]] = dataset.mapPartitions { sentenceIter =>
  // Each sentence will map to 0 or more Array[Int]
  sentenceIter.flatMap { sentence =>
    // Sentence of words, some of which map to a word index
    val wordIndexes = sentence.flatMap(bcVocabHash.value.get)
    // break wordIndexes into trunks of maxSentenceLength when has more
    wordIndexes.grouped(maxSentenceLength).map(_.toArray)
  }
}
```

- 为什么要映射成index？ 因为index更容易找到该word在huffman树中的位置。而且这一步还顺便把语料切成一段段。


### 2.2 syn0 和syn1 初始化

```
val syn0Global = Array.fill[Float](vocabSize * vectorSize)((initRandom.nextFloat() - 0.5f) / vectorSize)
val syn1Global = new Array[Float](vocabSize * vectorSize)
```
- syn0Global表示为叶子节点，syn1Global为非叶子节点，两者长度一样，均为 vocabSize * vectorSize。syn0Global用随机数初始化，syn1Global初始化为0向量。


### 2.3 迭代更新

```
val partial = newSentences.mapPartitionsWithIndex { case (idx, iter) =>
  val random = new XORShiftRandom(seed ^ ((idx + 1) << 16) ^ ((-k - 1) << 8))
  val syn0Modify = new Array[Int](vocabSize) //用于记录需要更新的词向量index
  val syn1Modify = new Array[Int](vocabSize)
  val model = iter.foldLeft((bcSyn0Global.value, bcSyn1Global.value, 0L, 0L)) {
    // model得到该分区的所有更新
    case ((syn0, syn1, lastWordCount, wordCount), sentence) =>
      var lwc = lastWordCount
      var wc = wordCount
      if (wordCount - lastWordCount > 10000) {
        lwc = wordCount
        // TODO: discount by iteration?
        alpha =
          learningRate * (1 - numPartitions * wordCount.toDouble / (trainWordsCount + 1))
        if (alpha < learningRate * 0.0001) alpha = learningRate * 0.0001
        logInfo("wordCount = " + wordCount + ", alpha = " + alpha)
      }
      wc += sentence.length
      var pos = 0
      while (pos < sentence.length) { // 这个循环是作为目标词遍历所有词
        val word = sentence(pos)
        val b = random.nextInt(window) //b为了随机找word的context
        // Train Skip-gram
        var a = b
        while (a < window * 2 + 1 - b) {
          if (a != window) {
            val c = pos - window + a //c就是用来确定的context词的index，而且确切来说是上文
            if (c >= 0 && c < sentence.length) {
              val lastWord = sentence(c) //lastword是context词
              val l1 = lastWord * vectorSize  //l1标识context词
              val neu1e = new Array[Float](vectorSize)
              // Hierarchical softmax
              var d = 0
              while (d < bcVocab.value(word).codeLen) {
                val inner = bcVocab.value(word).point(d)
                val l2 = inner * vectorSize // l2标识目标词经过的路径上的非叶子节点
                // Propagate hidden -> output
                var f = blas.sdot(vectorSize, syn0, l1, 1, syn1, l2, 1)
                //f=syn0[l1:l1+vectorSize-1] 点乘 syn1[l2:l2+vectorSize-1]
                if (f > -MAX_EXP && f < MAX_EXP) {
                  val ind = ((f + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2.0)).toInt
                  f = expTable.value(ind)
                  val g = ((1 - bcVocab.value(word).code(d) - f) * alpha).toFloat
                  blas.saxpy(vectorSize, g, syn1, l2, 1, neu1e, 0, 1) //neu1e = g*syn1 +neu1e
                  blas.saxpy(vectorSize, g, syn0, l1, 1, syn1, l2, 1) //syn1 = g*syn0 +syn1
                  syn1Modify(inner) += 1 //标识更改过的非叶子节点的index
                }
                d += 1
              }
              blas.saxpy(vectorSize, 1.0f, neu1e, 0, 1, syn0, l1, 1) // syn0=syn0+neu1e
              syn0Modify(lastWord) += 1
            }
          }
          a += 1
        }
        pos += 1
      }
      (syn0, syn1, lwc, wc)
  }
  val syn0Local = model._1
  val syn1Local = model._2
  // Only output modified vectors.
  Iterator.tabulate(vocabSize) { index =>
    if (syn0Modify(index) > 0) {
      Some((index, syn0Local.slice(index * vectorSize, (index + 1) * vectorSize)))
    } else {
      None
    }
  }.flatten ++ Iterator.tabulate(vocabSize) { index =>
    if (syn1Modify(index) > 0) {
      Some((index + vocabSize, syn1Local.slice(index * vectorSize, (index + 1) * vectorSize)))
    } else {
      None
    }
  }.flatten
}
```

- 以上是将整个文章放到不同分区中，每个executor对分区数据并行训练的结果。输入为上一轮得到的词向量（包括叶子节点和非叶子节点），输出为更新后的词向量。注意到最后(//only output modified vectors)输出的处理，通过syn0Modify和syn1Modify记录更改过的向量index,没有更改过的向量给0

- 这部分核心更新向量的代码实际上与skip-gram原来算法不一样，有两点区别：1. 原算法在每个窗口内，更新上下文的词向量和目标词路径上的非叶子节点向量，而这里是更新上下文词的非叶子节点向量和目标词的词向量。2. 原算法选定目标词后，遍历其窗口内的所有上下文词，而这里选定目标词后，只遍历目标词前面窗口的词，而且随机选择起始点。第1点区别影响的只是遍历的顺序，问题不大。第2点减少了计算的次数，如果样本量足够，影响也不大。


### 2.4 汇总executor的训练结果

```
val synAgg = partial.reduceByKey { case (v1, v2) =>
  blas.saxpy(vectorSize, 1.0f, v2, 1, v1, 1) // 把每个相同index的词向量加起来
  v1
}.collect() // collect到driver, Array[(Int,Array[Float])],前面是index,后面是词向量
var i = 0
while (i < synAgg.length) { // 把syn0Agg更新到syn0Global
  val index = synAgg(i)._1
  if (index < vocabSize) {
    Array.copy(synAgg(i)._2, 0, syn0Global, index * vectorSize, vectorSize)
  } else {
    Array.copy(synAgg(i)._2, 0, syn1Global, (index - vocabSize) * vectorSize, vectorSize)
  }
  i += 1
}
```

- 每个分区得到的都是一个新的词向量，汇总分区训练结果的做法是将所有新的词向量加一起来替代原来的词向量。这里可能跟我们常见的更新梯度有点不一样，但实际对结果影响不大（仅仅多加了若干个初始化的向量而已）。\r\n
&emsp;&emsp;word2vec的核心在于Huffman树以及训练计算叶节点与非叶节点的参数做梯度更新，过程还是比较容易理解的。但是，当词向量非常大的时候，需要将Huffman树在driver中生成并且下发到每个executor中并且加载在内存中，这就有点难为Spark了。这也就是为什么对于大规模参数的学习训练，大家更倾向于深度学习或者参数服务器了。
