---
title: 'Graph Convolution Neural Network For Weakly Supervised Abnormality Localization In Long Capsule Endoscopy Videos'

# Authors
# If you created a profile for a user (e.g. the default `admin` user), write the username (folder name) here
# and it will be replaced with their full name and linked to their profile.
authors:
  - Sodiq Adewole
  - Philip Fernandes
  - James Jablonski
  - Andrew Copland
  - Michael Porter
  - Sana Syed
  - Donald Brown

# Author notes (optional)
author_notes:
  - 'Equal contribution'
  - 'Equal contribution'

date: '2021-10-18T00:00:00Z'
doi: ''

# Schedule page publish date (NOT publication's date).
publishDate: '2021-10-18T00:00:00Z'

# Publication type.
# Legend: 0 = Uncategorized; 1 = Conference paper; 2 = Journal article;
# 3 = Preprint / Working Paper; 4 = Report; 5 = Book; 6 = Book section;
# 7 = Thesis; 8 = Patent
publication_types: ['1']

# Publication name and optional abbreviated publication name.
publication: In *IEEE International Conference on Big Data*
publication_short: In *IEEE*

abstract: Temporal activity localization in long videos is an important problem. The cost of obtaining frame level label for long Wireless Capsule Endoscopy (WCE) videos is prohibitive. In this paper, we propose an end-to-end temporal abnormality localization for long WCE videos using only weak video level labels. Physicians use Capsule Endoscopy (CE) as a non-surgical and non-invasive method to examine the entire digestive tract in order to diagnose diseases or abnormalities. While CE has revolutionized traditional endoscopy procedures, a single CE examination could last up to 8 hours generating as much as 100,000 frames. Physicians must review the entire video, frame-by-frame, in order to identify the frames capturing relevant abnormality. This, sometimes could be as few as just a single frame. Given this very high level of redundancy, analyzing long CE videos can be very tedious, time consuming and also error prone. This paper presents a novel multi-step method for an end-to-end localization of target frames capturing abnormalities of interest in the long video using only weak video labels. First we developed an automatic temporal segmentation using change point detection technique to temporally segment the video into uniform, homogeneous and identifiable segments. Then we employed Graph Convolutional Neural Network (GCNN) to learn a representation of each video segment. Using weak video segment labels, we trained our GCNN model to recognize each video segment as abnormal if it contains at least a single abnormal frame. Finally, leveraging the parameters of the trained GCNN model, we replaced the final layer of the network with a temporal pool layer to localize the relevant abnormal frames within each abnormal video segment. Our method achieved an accuracy of 89.9\% on the graph classification task and a specificity of 97.5\% on the abnormal frames localization task.

# Summary. An optional shortened abstract.
summary: Graph Convolution Neural Network, Wireless Capsule Endoscopy, Weakly Supervised Localization, Video Temporal Segmentation, Graph Classificatio

tags: []

# Display this page in the Featured widget?
featured: true

# Custom links (uncomment lines below)
# links:
# - name: Custom Link
#   url: http://example.org

url_pdf: 'https://arxiv.org/abs/2110.09110'
#url_code: 'https://github.com/wowchemy/wowchemy-hugo-themes'
# url_dataset: 'https://github.com/wowchemy/wowchemy-hugo-themes'
# url_poster: ''
# url_project: ''
# url_slides: ''
# url_source: 'https://github.com/wowchemy/wowchemy-hugo-themes'
# url_video: 'https://youtube.com'

# Featured image
# To use, add an image named `featured.jpg/png` to your page's folder.
image:
  caption: 'Image credit: [**Unsplash**](https://unsplash.com/photos/pLCdAaMFLTE)'
  focal_point: ''
  preview_only: false

# Associated Projects (optional).
#   Associate this publication with one or more of your projects.
#   Simply enter your project's folder or file name without extension.
#   E.g. `internal-project` references `content/project/internal-project/index.md`.
#   Otherwise, set `projects: []`.
projects:
  - Medical Image Analysis

# Slides (optional).
#   Associate this publication with Markdown slides.
#   Simply enter your slide deck's filename without extension.
#   E.g. `slides: "example"` references `content/slides/example/index.md`.
#   Otherwise, set `slides: ""`.
slides: example
---

{{% callout note %}}
Click the _Cite_ button above to demo the feature to enable visitors to import publication metadata into their reference management software.
{{% /callout %}}

{{% callout note %}}
Create your slides in Markdown - click the _Slides_ button to check out the example.
{{% /callout %}}

Supplementary notes can be added here, including [code, math, and images](https://wowchemy.com/docs/writing-markdown-latex/).
