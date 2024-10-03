# Deep Learning for Zero-day Malware Detection and Classification: A Survey Read

> Author: Hayden Chang 張皓鈞 B11030202

## Reference

[F. Deldar and M. Abadi, "Deep Learning for Zero-day Malware Detection and Classification: A Survey," ACM Computing Surveys, vol. 56, no. 2, Article 36, pp. 1-37, Feb. 2024](https://dl.acm.org/doi/abs/10.1145/3605775)



### 1. What is zero-day malware?

Zero-day malware refers to malicious software that has just emerged and hasn't been discovered or identified by anyone yet. These types of software exploit vulnerabilities in systems, and since there are no defenses developed against them, they are particularly dangerous.



### 2. What are unsupervised learning, semi-supervised learning, few-shot learning, and adversarial learning?

- **Unsupervised Learning**: This is a machine learning method that allows the model to find patterns from unlabeled data. It's like trying to figure out which objects in a dark room are chairs and which are tables, but without anyone telling you.
- **Semi-Supervised Learning**: This method combines labeled and unlabeled data. You have some known data (like a few labeled samples) along with a lot of unlabeled data. This helps the model learn better, similar to how a teacher gives you a few books, but there’s also a lot of material for you to explore on your own.
- **Few-Shot Learning**: This type of learning allows the model to recognize new things even when there are only a few samples available. For example, if you show it just one picture of a cat, it learns to identify cats without needing many pictures.
- **Adversarial Learning**: This is a method used to make models stronger. By adding some examples specifically designed to fool the model, it learns to resist these attacks, enhancing its ability to identify malicious software.



### 3. According to this paper, how can these learning methods help defend zero-day malware?

These learning methods help combat zero-day malware in the following ways:

- **Unsupervised Learning**: It can uncover unknown malicious behaviors, making it particularly effective against newly emerging malware.
- **Semi-Supervised Learning**: By combining a small amount of labeled data with a large amount of unlabeled data, it can help the model accurately detect unknown malware.
- **Few-Shot Learning**: It enables the model to identify new malware even with very few samples available, which is useful for new attacks.
- **Adversarial Learning**: It helps strengthen models against various malicious attacks, allowing them to detect a wider range of known and unknown malware, making them better suited for combating zero-day malware.



### 4. What is the difference between Malware Detection (MD) and Malware Classification (MC)?

- **Malware Detection (MD)**: This simply determines whether a sample is malicious or not, usually resulting in a binary answer: yes or no.
- **Malware Classification (MC)**: Not only does it identify if a sample is malicious, but it also categorizes it into types, such as whether it’s a virus, Trojan, or worm, which helps in understanding the characteristics of that malware better.



### 5. In malware classification, there are Category (C) or Family (F) classification. What is the difference between them?

- **Category Classification (C)**: This categorizes malware into broad types, such as all viruses.
- **Family Classification (F)**: This categorizes malware into specific families that share similar behaviors or code features. It’s like saying all cats are cats, but different cats can be categorized into different breeds.



### 6. In Few-Shot Learning, what are the support set and query set? What is the meaning of N-way K-shot?

- **Support Set**: This is the labeled samples used by the model for learning, helping it understand the characteristics of each category.
- **Query Set**: This is the set of samples used to test the model, which will classify these samples based on the support set.
- **N-way K-shot**: This is a setting that means there are K samples per category, with a total of N categories. This setup helps the model effectively classify with limited samples.



### 7. What are One-Shot Learning and Zero-Shot Learning?

- **One-Shot Learning**: This allows the model to learn to recognize a new category with only one sample. It’s like you can identify a cat just by having seen one picture of it.
- **Zero-Shot Learning**: In this case, the model has never seen a sample of the new category but can rely on previously learned knowledge to identify it. It’s like knowing about an animal you’ve never seen before just by hearing someone describe it, so you can guess what it is.
