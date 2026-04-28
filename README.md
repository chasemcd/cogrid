# Repository Coverage

[Full report](https://htmlpreview.github.io/?https://github.com/chasemcd/cogrid/blob/python-coverage-comment-action-data/htmlcov/index.html)

| Name                                                        |    Stmts |     Miss |   Cover |   Missing |
|------------------------------------------------------------ | -------: | -------: | ------: | --------: |
| cogrid/\_\_init\_\_.py                                      |        4 |        2 |     50% |     11-13 |
| cogrid/backend/\_\_init\_\_.py                              |        7 |        1 |     86% |        27 |
| cogrid/backend/\_dispatch.py                                |       32 |       10 |     69% |35, 50, 57-69, 82 |
| cogrid/backend/array\_ops.py                                |       22 |        6 |     73% |17, 35-37, 49-53 |
| cogrid/backend/env\_state.py                                |       50 |       32 |     36% |100-106, 114-117, 129-134, 139-141, 150-156, 163-201 |
| cogrid/backend/state\_view.py                               |       16 |        5 |     69% |     72-78 |
| cogrid/cogrid\_env.py                                       |      514 |      245 |     52% |72, 121, 125, 162-171, 176-188, 218-243, 265, 272-274, 284-287, 298, 304, 314, 333-334, 376-378, 400, 448-450, 466-471, 489-491, 512, 523-524, 554-562, 576-588, 596-625, 633-681, 692-747, 765-769, 777-781, 802, 806, 810, 822, 830-837, 846-853, 876-877, 881, 888, 892-935, 939-971, 976-991, 995-996, 1000, 1017-1018 |
| cogrid/constants.py                                         |        7 |        0 |    100% |           |
| cogrid/core/actions.py                                      |       36 |        0 |    100% |           |
| cogrid/core/agent.py                                        |       53 |        7 |     87% |27-29, 33, 54-55, 113 |
| cogrid/core/autowire.py                                     |      153 |       22 |     86% |144, 150, 158-163, 172, 177, 191-197, 209, 230, 239-240, 326-328 |
| cogrid/core/component\_registry.py                          |      120 |        5 |     96% |118, 142, 213, 223, 340 |
| cogrid/core/constants.py                                    |       24 |        0 |    100% |           |
| cogrid/core/directions.py                                   |        6 |        0 |    100% |           |
| cogrid/core/features.py                                     |       71 |        4 |     94% |83, 125, 152, 221 |
| cogrid/core/grid/\_\_init\_\_.py                            |        4 |        0 |    100% |           |
| cogrid/core/grid/grid.py                                    |      144 |       90 |     38% |38-42, 74-78, 88-90, 100, 108, 157-160, 180-183, 199-202, 227-268, 284-312, 321-355, 374-375, 389-393 |
| cogrid/core/grid/layouts.py                                 |       10 |        1 |     90% |        23 |
| cogrid/core/grid/parser.py                                  |       74 |       66 |     11% |49, 58-62, 86-182 |
| cogrid/core/grid/utils.py                                   |       37 |        4 |     89% |22-23, 73-74 |
| cogrid/core/objects/\_\_init\_\_.py                         |        6 |        0 |    100% |           |
| cogrid/core/objects/base.py                                 |      101 |       36 |     64% |52-54, 62, 81, 85, 99, 142-175, 180-205, 218, 220, 223-226 |
| cogrid/core/objects/builtins.py                             |       92 |       45 |     51% |48, 70-73, 81-94, 109, 114-122, 134-136, 141-153, 157-180 |
| cogrid/core/objects/containers.py                           |       75 |       27 |     64% |89-91, 149-179 |
| cogrid/core/objects/registry.py                             |      187 |       22 |     88% |44, 47, 55, 65, 73, 79, 142, 251-259, 329, 337, 353, 371, 376, 391, 446-455 |
| cogrid/core/objects/when.py                                 |       25 |        4 |     84% |     65-68 |
| cogrid/core/pipeline/\_\_init\_\_.py                        |        5 |        0 |    100% |           |
| cogrid/core/pipeline/context.py                             |       67 |       21 |     69% |98-100, 144-150, 165, 174-176, 185-187, 205-211 |
| cogrid/core/pipeline/interactions.py                        |      314 |       55 |     82% |239-240, 326, 370-384, 468-477, 486-487, 534-550, 611, 616-617, 745, 750-762 |
| cogrid/core/pipeline/movement.py                            |       60 |        2 |     97% |     46-47 |
| cogrid/core/pipeline/rewards.py                             |       63 |        3 |     95% |174, 189, 198 |
| cogrid/core/pipeline/step.py                                |      106 |        9 |     92% |52-58, 74-76, 83-85 |
| cogrid/core/typing.py                                       |       13 |        1 |     92% |        17 |
| cogrid/envs/\_\_init\_\_.py                                 |      122 |        5 |     96% |424-427, 454-455 |
| cogrid/envs/goal\_seeking/agent.py                          |       14 |       14 |      0% |      3-37 |
| cogrid/envs/goal\_seeking/goal\_seeking.py                  |       41 |       41 |      0% |      3-85 |
| cogrid/envs/overcooked/agent.py                             |        2 |        0 |    100% |           |
| cogrid/envs/overcooked/config.py                            |      162 |      116 |     28% |31-34, 67-70, 88-143, 175-207, 239-271, 308-369, 399-414, 432-445 |
| cogrid/envs/overcooked/features.py                          |      507 |       41 |     92% |358, 401, 406-407, 605-620, 870, 874-877, 892-893, 902, 906-907, 909, 956, 1001-1010, 1021-1024, 1043, 1056-1073, 1082-1091 |
| cogrid/envs/overcooked/overcooked\_grid\_objects.py         |      151 |       41 |     73% |29, 33, 46, 50, 67-69, 140-173, 181-184, 260, 263, 296, 300, 327, 332-335, 352, 357-360 |
| cogrid/envs/overcooked/recipes.py                           |       67 |        5 |     93% |67, 74, 77, 81, 98 |
| cogrid/envs/overcooked/rewards.py                           |      346 |      259 |     25% |122, 165-213, 314-371, 394-440, 464-476, 515-566, 590-603, 636-725, 751-842 |
| cogrid/envs/overcooked/test\_interactions.py                |      986 |      986 |      0% |   12-2218 |
| cogrid/envs/overcooked/test\_order\_observation.py          |      200 |      200 |      0% |    11-359 |
| cogrid/envs/overcooked/test\_state\_snapshot.py             |       56 |       56 |      0% |    17-116 |
| cogrid/envs/overcooked/test\_v2\_observation\_channels.py   |      395 |      395 |      0% |    11-799 |
| cogrid/envs/overcooked/test\_v2\_system.py                  |      326 |      326 |      0% |    17-738 |
| cogrid/envs/overcooked/v2\_objects.py                       |      116 |      116 |      0% |     9-295 |
| cogrid/envs/registry.py                                     |       10 |        2 |     80% |    11, 13 |
| cogrid/envs/search\_rescue/search\_rescue\_grid\_objects.py |       71 |       22 |     69% |25, 30-32, 45, 50-66, 78, 82-84, 96, 100, 112, 116, 128, 132, 144, 148 |
| cogrid/envs/search\_rescue/sr\_utils.py                     |       47 |       47 |      0% |     3-113 |
| cogrid/feature\_space/\_\_init\_\_.py                       |        1 |        0 |    100% |           |
| cogrid/feature\_space/features.py                           |       81 |       19 |     77% |     49-77 |
| cogrid/feature\_space/local\_view.py                        |      128 |       10 |     92% |125-126, 130, 235, 245, 255-262 |
| cogrid/rendering/\_\_init\_\_.py                            |        2 |        0 |    100% |           |
| cogrid/rendering/env\_renderer.py                           |       70 |       61 |     13% |12, 32-38, 66-133, 137-142 |
| cogrid/run\_interactive.py                                  |      108 |      108 |      0% |     3-231 |
| cogrid/test\_overcooked\_env.py                             |      102 |      102 |      0% |     1-240 |
| cogrid/visualization/rendering.py                           |       84 |       69 |     18% |15-22, 27-34, 40-49, 54-81, 87-90, 96-99, 104-128, 133-135, 148-153 |
| **TOTAL**                                                   | **6693** | **3766** | **44%** |           |

5 empty files skipped.


## Setup coverage badge

Below are examples of the badges you can use in your main branch `README` file.

### Direct image

[![Coverage badge](https://raw.githubusercontent.com/chasemcd/cogrid/python-coverage-comment-action-data/badge.svg)](https://htmlpreview.github.io/?https://github.com/chasemcd/cogrid/blob/python-coverage-comment-action-data/htmlcov/index.html)

This is the one to use if your repository is private or if you don't want to customize anything.

### [Shields.io](https://shields.io) Json Endpoint

[![Coverage badge](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/chasemcd/cogrid/python-coverage-comment-action-data/endpoint.json)](https://htmlpreview.github.io/?https://github.com/chasemcd/cogrid/blob/python-coverage-comment-action-data/htmlcov/index.html)

Using this one will allow you to [customize](https://shields.io/endpoint) the look of your badge.
It won't work with private repositories. It won't be refreshed more than once per five minutes.

### [Shields.io](https://shields.io) Dynamic Badge

[![Coverage badge](https://img.shields.io/badge/dynamic/json?color=brightgreen&label=coverage&query=%24.message&url=https%3A%2F%2Fraw.githubusercontent.com%2Fchasemcd%2Fcogrid%2Fpython-coverage-comment-action-data%2Fendpoint.json)](https://htmlpreview.github.io/?https://github.com/chasemcd/cogrid/blob/python-coverage-comment-action-data/htmlcov/index.html)

This one will always be the same color. It won't work for private repos. I'm not even sure why we included it.

## What is that?

This branch is part of the
[python-coverage-comment-action](https://github.com/marketplace/actions/python-coverage-comment)
GitHub Action. All the files in this branch are automatically generated and may be
overwritten at any moment.