# CLIP music video
Code for making music videos using CLIP

## How to run (using biggan)
```console
 python3 main.py --epochs 400 --textfile yannic_lyrics.txt --audiofile Weasle_sample_audio.mp3
 ```
 use less epochs to trade off quality and speed.

## References

[https://twitter.com/advadnoun](https://twitter.com/advadnoun)

[https://towardsdatascience.com/generating-images-from-prompts-using-clip-and-stylegan-1f9ed495ddda](https://towardsdatascience.com/generating-images-from-prompts-using-clip-and-stylegan-1f9ed495ddda)

[https://www.reddit.com/r/MachineLearning/comments/lemeyo/p_generate_faces_from_prompts_using_clip_and/](https://www.reddit.com/r/MachineLearning/comments/lemeyo/p_generate_faces_from_prompts_using_clip_and/)

[https://arxiv.org/pdf/2102.01645v2.pdf](https://arxiv.org/pdf/2102.01645v2.pdf)

[https://colab.research.google.com/drive/1FoHdqoqKntliaQKnMoNs3yn5EALqWtvP?usp=sharing](https://colab.research.google.com/drive/1FoHdqoqKntliaQKnMoNs3yn5EALqWtvP?usp=sharing)

[https://colab.research.google.com/drive/1NCceX2mbiKOSlAd_o7IU7nA9UskKN5WR?usp=sharing](https://colab.research.google.com/drive/1NCceX2mbiKOSlAd_o7IU7nA9UskKN5WR?usp=sharing)

[https://colab.research.google.com/drive/1fWka_U56NhCegbbrQPt4PWpHPtNRdU49?usp=sharing#scrollTo=zvZFRZtcv8Mp](https://colab.research.google.com/drive/1fWka_U56NhCegbbrQPt4PWpHPtNRdU49?usp=sharing#scrollTo=zvZFRZtcv8Mp)

[https://deepai.org/publication/generating-images-from-caption-and-vice-versa-via-clip-guided-generative-latent-space-search](https://deepai.org/publication/generating-images-from-caption-and-vice-versa-via-clip-guided-generative-latent-space-search)

[https://www.reddit.com/r/MachineLearning/comments/kzr4mg/p_the_big_sleep_texttoimage_generation_using/?user_id=232319233763](https://www.reddit.com/r/MachineLearning/comments/kzr4mg/p_the_big_sleep_texttoimage_generation_using/?user_id=232319233763)

[https://www.reddit.com/r/MachineLearning/comments/ky8fq8/p_a_colab_notebook_from_ryan_murdock_that_creates/](https://www.reddit.com/r/MachineLearning/comments/ky8fq8/p_a_colab_notebook_from_ryan_murdock_that_creates/)

[https://twitter.com/jonathanfly/status/1362396180978798593](https://twitter.com/jonathanfly/status/1362396180978798593)

[https://github.com/orpatashnik/StyleCLIP](https://github.com/orpatashnik/StyleCLIP)

[https://twitter.com/Miles_Brundage/status/1363268217385349121](https://twitter.com/Miles_Brundage/status/1363268217385349121)

[https://twitter.com/jonathanfly/status/1363550050899554311](https://twitter.com/jonathanfly/status/1363550050899554311)

[https://www.reddit.com/r/MachineLearning/comments/ldc6oc/p_list_of_sitesprogramsprojects_that_use_openais/](https://www.reddit.com/r/MachineLearning/comments/ldc6oc/p_list_of_sitesprogramsprojects_that_use_openais/)

[https://colab.research.google.com/drive/1Q-TbYvASMPRMXCOQjkxxf72CXYjR_8Vp?usp=sharing](https://colab.research.google.com/drive/1Q-TbYvASMPRMXCOQjkxxf72CXYjR_8Vp?usp=sharing)

[https://gist.github.com/l4rz/7040835c3f8266d8b8ea3615a0b49494](https://gist.github.com/l4rz/7040835c3f8266d8b8ea3615a0b49494)

[https://twitter.com/eps696/status/1363419772252278791](https://twitter.com/eps696/status/1363419772252278791)

# install ImageMagick for Moviepy textclips
instructions here 
```console 
https://techpiezo.com/linux/install-imagemagick-in-ubuntu-20-04-lts/ 
```
To edit permission follow this steps:
```console 
file /etc/ImageMagick/policy.xml here,
changed <policy domain="path" rights="none" pattern="@*" />
to <policy domain="path" rights="read,write" pattern="@*" />
```

# BigGan_Dall-E_Clip with moviepy
I used the Dall-E Encoder: https://colab.research.google.com/drive/1Q-TbYvASMPRMXCOQjkxxf72CXYjR_8Vp?usp=sharing#scrollTo=7EuUz-ICNKUr
Please create a folder 'output' for the generated images to be saved in.
Please install clip and dall-e
```console
pip install git+https://github.com/openai/CLIP.git
pip install DALL-E
```
the dall-e decoder is here:
```console
https://cdn.openai.com/dall-e/decoder.pkl
```
