1. 本工程实现了以Transformer作为编码器，以Pointer-Net作为解码器的结构，主要目的在于从输入中得到关注对象的索引。
2. 本工程修改自 tensorflow.contrib.seq2seq 中的各组件：helper.py、basic_decoder.py、decoder.py和attention_wrapper.py
3. helper.py 所实现的功能最为简单，目前所用的是 GreedyHelper，实现了在一步解码之后得到其嵌入表示，将其作为输入传到下一个解码步。
4. attention_wrapper.py 实现了对标准RNNCell类的新封装，在原始RNNCell的输出基础上加上了注意力机制，其中主要方法为prepare_memory：主要在于
根据传入的掩码对memory作掩码，遮盖掉那些已经不存在的单位或者是<pad>所在的位置，_maybe_mask_score提供了在计算alignments(分配系数)的时候的
掩码，能保证选不到某些单词或者单位。通过attention_wrapper_state类传递Cell的内部状态，可以根据自己需要定制，如果有额外需要保存的信息，可以通
过改此类在不同解码步之间传递。有这种需求的尽量不要修改call()方法的参数，因为这会造成和原有的RNNCell的call()方法的冲突，能通过修改
attention_wrapper_state类得到解决尽量这样做。
5. basic_decoder 和 decoder 中的 dynamic_decode 方法联合实现了动态解码的过程。dynamic_decode 方法通过使用while_loop循环每次调用
basic_decoder 中的step()方法，实现动态解码，basic_decoder 中的 step() 方法也相当之简单，主要在于调用了 GreedyHelper 类中的sample_id方法，
此方法得到每次解码步中的 id 的采样，也就是每一个解码步得到的在单词表中的索引，以及调用了attention_wrapper的 call() 方法，得到结合了注意力机制
的 RNNCell 的输出。