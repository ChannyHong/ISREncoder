<style TYPE="text/css">
code.has-jax {font: inherit; font-size: 100%; background: inherit; border: inherit;}
</style>
<script type="text/x-mathjax-config">
MathJax.Hub.Config({
    tex2jax: {
        inlineMath: [['$','$'], ['\\(','\\)']],
        skipTags: ['script', 'noscript', 'style', 'textarea', 'pre'] // removed 'code' entry
    }
});
MathJax.Hub.Queue(function() {
    var all = MathJax.Hub.getAllJax(), i;
    for(i = 0; i < all.length; i += 1) {
        all[i].SourceElement().parentNode.className += ' has-jax';
    }
});
</script>
<script type="text/javascript" src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.4/MathJax.js?config=TeX-AMS_HTML-full"></script>

# ISREncoder

This is the project page for the Tensorflow implementation of the paper, "Unsupervised Interlingual Semantic Representations from Sentence Embeddings for Zero-Shot Cross-Lingual Transfer", accepted for presentation at the Thirty-Fourth AAAI Conference on Artificial Intelligence (AAAI-20).

The full implementation will be released alongside the poster presentation at AAAI-20 in early Februrary.



## Abstract

As numerous modern NLP models demonstrate high-performance in various tasks when trained with resource-rich language data sets such as those of English, there has been a shift in attention to the idea of applying such learning to low-resource languages via zero-shot or few-shot cross-lingual transfer. While the most prominent efforts made previously on achieving this feat entails the use of parallel corpora for sentence alignment training, we seek to generalize further by assuming plausible scenarios in which such parallel data sets are unavailable. In this work, we present a novel architecture for training interlingual semantic representations on top of sentence embeddings in a completely unsupervised manner, and demonstrate its effectiveness in zero-shot cross-lingual transfer in natural language inference task. Furthermore, we showcase a method of leveraging this framework in a few-shot scenario, and finally analyze the distributional and permutational alignment across languages of these interlingual semantic representations.

## Model

<img src="https://github.com/ChannyHong/ISREncoder/blob/master/imgs/training_flow.png" width="1000px"/>

## Main Results

Model Type | en | es | de | zh | ar
---------- | :------: | :------: | :------: | :------: | :------:
BSE (Baseline) | 63.8 | 57.1 | 51.9 | 53.4 | 50.2
ISR ($\lambda_{isr} = 0$) | 65.2 | 57.9 | 55.0 | 55.8 | 50.4
ISR ($\lambda_{D,G_{cls}} = 0$) | 60.1 | 56.1 | 52.6 | 51.2 | 50.0
ISR ($\lambda_{rec} = 0$) | 37.6 | 36.3 | 36.0 | 38.0 | 37.4
**ISR** | **65.4** | **60.4** | **58.8** | **58.4** | **55.4**
   

