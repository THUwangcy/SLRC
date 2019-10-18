## Data

*The original data of Order dataset can be downloaded on [Google Drive](https://drive.google.com/drive/folders/1ZDjnC2L0pWpdqd5TNXMICaB_GOsZXTI8?usp=sharing) or [Baidu Cloud Drive](https://pan.baidu.com/s/1WTiDOURoF_X0TGqoQZBUCA&shfl=shareset) (access code: **mw5r**)*

Files under this folder are original data (like order.txt) in following format:

UserName \t ItemName \t Score \t Timestamp

The Score column has no use, just for catering to some other datasets' format.



The original data will be preprocessed by Preprocess.py, and corresponding dataset folder will be created (like ./data_order/). There are four files in each dataset folder:

book.csv:

- Total records aggregated by user, representing user consuming sequence in time-ascending order.

- Each line is in format: UserID \t [ (ItemID, Timestamp), (ItemID, Timestamp), ... ]

train.csv:

- Training instances. Note that here we save consumption **order** in book.csv. 
- Each line is in format: UserID \t ConsumptionOrder
- (e.g. '2 \t 0' means the first consumption of user_2,  and corresponding item_id can be found in book.csv)

dev.csv:

- Validating instances. There can be multiple ground truth items that are bought the same time. Ground truth items are ensured to be included in candidate list.
- Each line is in format: UserID \t [ ground truth consumption *orders* ] \t [ candidate item *ids* ]

test.csv:

- Testing instances. Similar with dev.csv.
- Each line is in format: UserID \t [ ground truth consumption *orders* ] \t [ candidate item *ids* ]



The scale of candidate items are relevant to the number of unique items in each dataset. See ./src/common/constants.py for concret settings.
