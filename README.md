# Repository Coverage

[Full report](https://htmlpreview.github.io/?https://github.com/chasemcd/cogrid/blob/python-coverage-comment-action-data/htmlcov/index.html)

| Name                                                        |    Stmts |     Miss |   Cover |   Missing |
|------------------------------------------------------------ | -------: | -------: | ------: | --------: |
| cogrid/\_\_init\_\_.py                                      |        5 |        3 |     40% |     11-14 |
| cogrid/backend/\_\_init\_\_.py                              |        7 |        1 |     86% |        27 |
| cogrid/backend/\_dispatch.py                                |       30 |        9 |     70% |35, 50, 57-69 |
| cogrid/backend/array\_ops.py                                |        9 |        1 |     89% |        17 |
| cogrid/backend/env\_state.py                                |       50 |       32 |     36% |100-106, 114-117, 129-134, 139-141, 150-156, 163-201 |
| cogrid/backend/state\_view.py                               |       16 |        7 |     56% |65-66, 72-78 |
| cogrid/cogrid\_env.py                                       |      464 |      211 |     55% |67, 117, 121, 168-191, 213, 220-222, 232-235, 246, 252, 262, 281-282, 324-326, 348, 390-392, 408-413, 431-433, 456, 467-468, 481-503, 511-555, 566-621, 636-640, 648-652, 673, 677, 681, 693, 701-708, 717-724, 747-749, 758-779, 783-808, 816-821, 827-861, 870-875, 879-908, 912-913, 917, 934-935 |
| cogrid/constants.py                                         |        7 |        0 |    100% |           |
| cogrid/core/actions.py                                      |       17 |        0 |    100% |           |
| cogrid/core/agent.py                                        |       73 |       17 |     77% |37-39, 43, 64-65, 69, 78, 82, 107-117, 149, 162-169 |
| cogrid/core/autowire.py                                     |      102 |       17 |     83% |113-120, 131-137, 157-158, 219-221 |
| cogrid/core/component\_registry.py                          |      121 |        4 |     97% |110, 244, 254, 365 |
| cogrid/core/constants.py                                    |       40 |        0 |    100% |           |
| cogrid/core/directions.py                                   |        6 |        0 |    100% |           |
| cogrid/core/features.py                                     |       49 |        2 |     96% |    62, 90 |
| cogrid/core/grid.py                                         |      211 |      151 |     28% |38-42, 74-78, 88-90, 100, 108, 141-143, 163-166, 186-189, 205-208, 212-231, 235-253, 278-315, 331-359, 368-402, 424-425, 439-443, 447-484 |
| cogrid/core/grid\_object.py                                 |        4 |        0 |    100% |           |
| cogrid/core/grid\_object\_base.py                           |      144 |       59 |     59% |66, 70, 78, 82, 86-87, 91-97, 101, 113, 117-119, 127, 143, 147, 155, 159, 163-165, 175, 218-222, 226-259, 264-289, 302, 304, 307-310 |
| cogrid/core/grid\_object\_registry.py                       |      105 |       16 |     85% |36, 39, 47, 57, 65, 71, 201-209, 228, 233, 248, 295-300 |
| cogrid/core/grid\_objects.py                                |      112 |       59 |     47% |38, 51, 57, 76, 80-83, 91-104, 119, 123, 128-136, 149-151, 155, 159, 163-171, 176-188, 192-215 |
| cogrid/core/grid\_utils.py                                  |       43 |        9 |     79% |22-23, 73-74, 81-86 |
| cogrid/core/interactions.py                                 |       48 |        0 |    100% |           |
| cogrid/core/layout\_parser.py                               |       74 |       74 |      0% |    29-182 |
| cogrid/core/layouts.py                                      |       10 |        1 |     90% |        23 |
| cogrid/core/movement.py                                     |       59 |        2 |     97% |     46-47 |
| cogrid/core/rewards.py                                      |        4 |        0 |    100% |           |
| cogrid/core/roles.py                                        |        5 |        0 |    100% |           |
| cogrid/core/step\_pipeline.py                               |       94 |        9 |     90% |52-58, 74-76, 83-85 |
| cogrid/core/typing.py                                       |        7 |        0 |    100% |           |
| cogrid/envs/\_\_init\_\_.py                                 |       39 |        2 |     95% |   146-155 |
| cogrid/envs/goal\_seeking/agent.py                          |       14 |       14 |      0% |      3-37 |
| cogrid/envs/goal\_seeking/goal\_seeking.py                  |       41 |       41 |      0% |      3-85 |
| cogrid/envs/overcooked/\_\_init\_\_.py                      |        1 |        0 |    100% |           |
| cogrid/envs/overcooked/agent.py                             |        8 |        3 |     62% |     22-27 |
| cogrid/envs/overcooked/config.py                            |      201 |       26 |     87% |135-137, 315, 324-351 |
| cogrid/envs/overcooked/features.py                          |      231 |        1 |     99% |       310 |
| cogrid/envs/overcooked/overcooked\_grid\_objects.py         |      210 |       86 |     59% |28, 35, 39, 56, 63, 67, 80, 84, 88-90, 103, 107, 111-113, 143, 150-157, 161-168, 172, 177, 181-185, 190, 194-204, 208-210, 255-281, 309, 313, 317-319, 336, 345, 349, 370-377, 381, 398, 405, 410-413, 434, 441, 446-449 |
| cogrid/envs/overcooked/rewards.py                           |       73 |        5 |     93% |71, 128-129, 171-172 |
| cogrid/envs/overcooked/test\_interactions.py                |      161 |      161 |      0% |    12-418 |
| cogrid/envs/registry.py                                     |        8 |        1 |     88% |        11 |
| cogrid/envs/search\_rescue/search\_rescue\_grid\_objects.py |      132 |       70 |     47% |23, 29, 34-36, 51, 57, 62-78, 93, 100, 104-119, 123-125, 140, 147-153, 157, 172, 179-185, 189, 204, 211-224, 228, 243-247, 251-253, 261-274, 278 |
| cogrid/envs/search\_rescue/sr\_utils.py                     |       46 |       46 |      0% |     3-112 |
| cogrid/feature\_space/\_\_init\_\_.py                       |        1 |        0 |    100% |           |
| cogrid/feature\_space/features.py                           |       81 |       19 |     77% |     49-77 |
| cogrid/rendering/\_\_init\_\_.py                            |        2 |        0 |    100% |           |
| cogrid/rendering/env\_renderer.py                           |       53 |       44 |     17% |12, 32-38, 61-106, 110-115 |
| cogrid/run\_interactive.py                                  |      110 |      110 |      0% |     3-243 |
| cogrid/test\_overcooked\_env.py                             |      118 |      118 |      0% |     1-275 |
| cogrid/visualization/rendering.py                           |       84 |       69 |     18% |15-22, 27-34, 40-49, 54-81, 87-90, 96-99, 104-128, 133-135, 148-153 |
| **TOTAL**                                                   | **3530** | **1500** | **58%** |           |

4 empty files skipped.


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