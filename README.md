# Repository Coverage

[Full report](https://htmlpreview.github.io/?https://github.com/chasemcd/cogrid/blob/python-coverage-comment-action-data/htmlcov/index.html)

| Name                                                        |    Stmts |     Miss |   Cover |   Missing |
|------------------------------------------------------------ | -------: | -------: | ------: | --------: |
| cogrid/\_\_init\_\_.py                                      |        5 |        3 |     40% |     11-14 |
| cogrid/backend/\_\_init\_\_.py                              |        7 |        1 |     86% |        27 |
| cogrid/backend/\_dispatch.py                                |       30 |        9 |     70% |35, 50, 57-69 |
| cogrid/backend/array\_ops.py                                |       17 |        4 |     76% | 17, 35-39 |
| cogrid/backend/env\_state.py                                |       50 |       32 |     36% |100-106, 114-117, 129-134, 139-141, 150-156, 163-201 |
| cogrid/backend/state\_view.py                               |       16 |        5 |     69% |     72-78 |
| cogrid/cogrid\_env.py                                       |      470 |      212 |     55% |67, 117, 121, 149, 178-201, 223, 230-232, 242-245, 256, 262, 272, 291-292, 334-336, 358, 401-403, 419-424, 442-444, 467, 478-479, 492-514, 522-566, 577-632, 647-651, 659-663, 684, 688, 692, 704, 712-719, 728-735, 758-760, 769-790, 794-819, 827-832, 838-872, 881-886, 890-919, 923-924, 928, 945-946 |
| cogrid/constants.py                                         |        7 |        0 |    100% |           |
| cogrid/core/actions.py                                      |       17 |        0 |    100% |           |
| cogrid/core/agent.py                                        |       69 |       15 |     78% |37-39, 43, 64-65, 69, 99-109, 141, 154-161 |
| cogrid/core/autowire.py                                     |      148 |       24 |     84% |147, 153, 161-166, 175, 180, 192-198, 203-204, 209, 235, 244-245, 327-329 |
| cogrid/core/component\_registry.py                          |       97 |        4 |     96% |103, 174, 184, 293 |
| cogrid/core/constants.py                                    |       40 |        0 |    100% |           |
| cogrid/core/containers.py                                   |      134 |       32 |     76% |99, 106, 109, 113, 130, 194-196, 254-284 |
| cogrid/core/directions.py                                   |        6 |        0 |    100% |           |
| cogrid/core/features.py                                     |       59 |        2 |     97% |   76, 112 |
| cogrid/core/grid.py                                         |      211 |      151 |     28% |38-42, 74-78, 88-90, 100, 108, 141-143, 163-166, 186-189, 205-208, 212-231, 235-253, 278-315, 331-359, 368-402, 424-425, 439-443, 447-484 |
| cogrid/core/grid\_object.py                                 |        6 |        0 |    100% |           |
| cogrid/core/grid\_object\_base.py                           |      116 |       44 |     62% |51, 59-61, 69, 85, 89, 97, 101, 111, 154-158, 162-195, 200-225, 238, 240, 243-246 |
| cogrid/core/grid\_object\_registry.py                       |      173 |       20 |     88% |36, 39, 47, 57, 65, 71, 134, 239-247, 313, 321, 337, 355, 360, 375, 422-427 |
| cogrid/core/grid\_objects.py                                |       96 |       47 |     51% |39, 52, 74-77, 85-98, 113, 118-126, 138-140, 144, 149-161, 165-188 |
| cogrid/core/grid\_utils.py                                  |       43 |        9 |     79% |22-23, 73-74, 81-86 |
| cogrid/core/interactions.py                                 |      274 |       41 |     85% |333-348, 412-416, 454-455, 497, 499, 508-509, 550-566, 624-626 |
| cogrid/core/layout\_parser.py                               |       74 |       74 |      0% |    29-182 |
| cogrid/core/layouts.py                                      |       10 |        1 |     90% |        23 |
| cogrid/core/movement.py                                     |       59 |        2 |     97% |     46-47 |
| cogrid/core/rewards.py                                      |       53 |        3 |     94% |122, 137, 146 |
| cogrid/core/roles.py                                        |        5 |        5 |      0% |      3-11 |
| cogrid/core/step\_pipeline.py                               |       94 |        9 |     90% |52-58, 74-76, 83-85 |
| cogrid/core/typing.py                                       |        7 |        0 |    100% |           |
| cogrid/core/when.py                                         |       25 |        4 |     84% |     65-68 |
| cogrid/envs/\_\_init\_\_.py                                 |       34 |        0 |    100% |           |
| cogrid/envs/goal\_seeking/agent.py                          |       14 |       14 |      0% |      3-37 |
| cogrid/envs/goal\_seeking/goal\_seeking.py                  |       41 |       41 |      0% |      3-85 |
| cogrid/envs/overcooked/\_\_init\_\_.py                      |        1 |        0 |    100% |           |
| cogrid/envs/overcooked/agent.py                             |        2 |        0 |    100% |           |
| cogrid/envs/overcooked/config.py                            |       62 |       62 |      0% |    15-173 |
| cogrid/envs/overcooked/features.py                          |      325 |        9 |     97% |357, 400, 405-406, 604-619 |
| cogrid/envs/overcooked/overcooked\_grid\_objects.py         |      116 |       25 |     78% |24, 28, 41, 45, 61-63, 112-122, 126-128, 186, 189, 222, 226, 253, 258-261, 278, 283-286 |
| cogrid/envs/overcooked/rewards.py                           |       89 |       28 |     69% |122, 151-190, 285-297 |
| cogrid/envs/overcooked/test\_interactions.py                |     1048 |     1048 |      0% |   12-2277 |
| cogrid/envs/registry.py                                     |        8 |        1 |     88% |        11 |
| cogrid/envs/search\_rescue/search\_rescue\_grid\_objects.py |       72 |       23 |     68% |24, 29-31, 44, 49-65, 77, 81, 85-87, 99, 103, 115, 119, 131, 135, 147, 151 |
| cogrid/envs/search\_rescue/sr\_utils.py                     |       46 |       46 |      0% |     3-112 |
| cogrid/feature\_space/\_\_init\_\_.py                       |        1 |        0 |    100% |           |
| cogrid/feature\_space/features.py                           |       81 |       19 |     77% |     49-77 |
| cogrid/rendering/\_\_init\_\_.py                            |        2 |        0 |    100% |           |
| cogrid/rendering/env\_renderer.py                           |       53 |       44 |     17% |12, 32-38, 61-106, 110-115 |
| cogrid/run\_interactive.py                                  |      110 |      110 |      0% |     3-243 |
| cogrid/test\_overcooked\_env.py                             |      102 |      102 |      0% |     1-240 |
| cogrid/visualization/rendering.py                           |       84 |       69 |     18% |15-22, 27-34, 40-49, 54-81, 87-90, 96-99, 104-128, 133-135, 148-153 |
| **TOTAL**                                                   | **4709** | **2394** | **49%** |           |

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