# Label quality check

Checked 12 training images (seeds 42 and 1).

| Image | Problem |
|---|---|
| agri_0_2158 | small seedlings without boxes |
| agri_0_6547 | edge plants without boxes |
| agri_0_6353 | edge plants without boxes |
| agri_0_2997 | edge plants without boxes |
| agri_0_601  | small plant in corner without box |
| agri_0_1017 | one loose box over several weeds; other weeds unlabeled |
| agri_0_1361 | one box for a whole cluster (vs. one box per seedling in agri_0_495) |
| agri_0_1494 | labeled weed, but appears to be the same species as crop plants (e.g. agri_0_2964) |
| agri_0_3647 | edge-stretching artifact from augmentation |

Good examples: agri_0_2964, agri_0_1024, agri_0_495.