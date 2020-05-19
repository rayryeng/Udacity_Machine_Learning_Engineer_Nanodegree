import torchvision.models as models
import torchvision.transforms as transforms
import torch.nn as nn
import torch

def get_labels():
    labels = ['001.Affenpinscher',
     '002.Afghan_hound',
     '003.Airedale_terrier',
     '004.Akita',
     '005.Alaskan_malamute',
     '006.American_eskimo_dog',
     '007.American_foxhound',
     '008.American_staffordshire_terrier',
     '009.American_water_spaniel',
     '010.Anatolian_shepherd_dog',
     '011.Australian_cattle_dog',
     '012.Australian_shepherd',
     '013.Australian_terrier',
     '014.Basenji',
     '015.Basset_hound',
     '016.Beagle',
     '017.Bearded_collie',
     '018.Beauceron',
     '019.Bedlington_terrier',
     '020.Belgian_malinois',
     '021.Belgian_sheepdog',
     '022.Belgian_tervuren',
     '023.Bernese_mountain_dog',
     '024.Bichon_frise',
     '025.Black_and_tan_coonhound',
     '026.Black_russian_terrier',
     '027.Bloodhound',
     '028.Bluetick_coonhound',
     '029.Border_collie',
     '030.Border_terrier',
     '031.Borzoi',
     '032.Boston_terrier',
     '033.Bouvier_des_flandres',
     '034.Boxer',
     '035.Boykin_spaniel',
     '036.Briard',
     '037.Brittany',
     '038.Brussels_griffon',
     '039.Bull_terrier',
     '040.Bulldog',
     '041.Bullmastiff',
     '042.Cairn_terrier',
     '043.Canaan_dog',
     '044.Cane_corso',
     '045.Cardigan_welsh_corgi',
     '046.Cavalier_king_charles_spaniel',
     '047.Chesapeake_bay_retriever',
     '048.Chihuahua',
     '049.Chinese_crested',
     '050.Chinese_shar-pei',
     '051.Chow_chow',
     '052.Clumber_spaniel',
     '053.Cocker_spaniel',
     '054.Collie',
     '055.Curly-coated_retriever',
     '056.Dachshund',
     '057.Dalmatian',
     '058.Dandie_dinmont_terrier',
     '059.Doberman_pinscher',
     '060.Dogue_de_bordeaux',
     '061.English_cocker_spaniel',
     '062.English_setter',
     '063.English_springer_spaniel',
     '064.English_toy_spaniel',
     '065.Entlebucher_mountain_dog',
     '066.Field_spaniel',
     '067.Finnish_spitz',
     '068.Flat-coated_retriever',
     '069.French_bulldog',
     '070.German_pinscher',
     '071.German_shepherd_dog',
     '072.German_shorthaired_pointer',
     '073.German_wirehaired_pointer',
     '074.Giant_schnauzer',
     '075.Glen_of_imaal_terrier',
     '076.Golden_retriever',
     '077.Gordon_setter',
     '078.Great_dane',
     '079.Great_pyrenees',
     '080.Greater_swiss_mountain_dog',
     '081.Greyhound',
     '082.Havanese',
     '083.Ibizan_hound',
     '084.Icelandic_sheepdog',
     '085.Irish_red_and_white_setter',
     '086.Irish_setter',
     '087.Irish_terrier',
     '088.Irish_water_spaniel',
     '089.Irish_wolfhound',
     '090.Italian_greyhound',
     '091.Japanese_chin',
     '092.Keeshond',
     '093.Kerry_blue_terrier',
     '094.Komondor',
     '095.Kuvasz',
     '096.Labrador_retriever',
     '097.Lakeland_terrier',
     '098.Leonberger',
     '099.Lhasa_apso',
     '100.Lowchen',
     '101.Maltese',
     '102.Manchester_terrier',
     '103.Mastiff',
     '104.Miniature_schnauzer',
     '105.Neapolitan_mastiff',
     '106.Newfoundland',
     '107.Norfolk_terrier',
     '108.Norwegian_buhund',
     '109.Norwegian_elkhound',
     '110.Norwegian_lundehund',
     '111.Norwich_terrier',
     '112.Nova_scotia_duck_tolling_retriever',
     '113.Old_english_sheepdog',
     '114.Otterhound',
     '115.Papillon',
     '116.Parson_russell_terrier',
     '117.Pekingese',
     '118.Pembroke_welsh_corgi',
     '119.Petit_basset_griffon_vendeen',
     '120.Pharaoh_hound',
     '121.Plott',
     '122.Pointer',
     '123.Pomeranian',
     '124.Poodle',
     '125.Portuguese_water_dog',
     '126.Saint_bernard',
     '127.Silky_terrier',
     '128.Smooth_fox_terrier',
     '129.Tibetan_mastiff',
     '130.Welsh_springer_spaniel',
     '131.Wirehaired_pointing_griffon',
     '132.Xoloitzcuintli',
     '133.Yorkshire_terrier']

    return labels

class Net(nn.Module):
    ### TODO: choose an architecture, and complete the class
    def __init__(self):
        super(Net, self).__init__()
        ## Define layers of a CNN

        # Define 3 Conv 2D layers with progressively larger
        # numbers of filters.  First two will have stride of 2
        # to prevent overfitting.  Last one has stride 1
        self.conv1 = nn.Conv2d(3, 32, 3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)

        # Max Pooling 2D after each Conv
        self.pool = nn.MaxPool2d(2, 2)

        # FC layers for non-linear mapping of features
        # to classifying dog
        self.fc1 = nn.Linear(7*7*128, 512)
        # One more for good measure
        self.fc2 = nn.Linear(512, 133)

        # Define dropout
        self.dropout = nn.Dropout(0.25)

    def forward(self, x):
        ## Define forward behavior
        # 3 x 224 x 224

        # CONV-RELU-POOL 1
        x = F.relu(self.conv1(x)) # 32 x 112 x 112
        x = self.pool(x) # 32 x 56 x 56

        # CONV-RELU-POOL 2
        x = F.relu(self.conv2(x)) # 64 x 28 x 28
        x = self.pool(x) # 64 x 14 x 14

        # CONV-RELU-POOL 3
        x = F.relu(self.conv3(x)) # 128 x 14 x 14 - note stride=1
        x = self.pool(x) # 128 x 7 x 7

        # Flatten
        x = x.view(x.size(0), -1)

        # DROPOUT-FC-RELU
        x = self.dropout(x)
        x = F.relu(self.fc1(x)) 

        # DROPOUT-FC
        x = self.dropout(x)
        x = self.fc2(x)

        # Return raw activations as we're using nn.CrossEntropyLoss
        return x

def create_dog_breed_classification_model_scratch():
    return Net()

def create_dog_breed_classification_model():
    ## TODO: Specify model architecture
    model_transfer = models.resnet50(pretrained=True)

    # We should only fine-tune the last layer
    # The conv layers for feature extraction are just fine
    for param in model_transfer.parameters():
        param.requires_grad = False

    # Replace the last fully connected layer with the total
    # number of classes of dogs
    num_ftrs = model_transfer.fc.in_features
    model_transfer.fc = nn.Linear(num_ftrs, 133)
    
    return model_transfer

def perform_inference(model_transfer, img):
    tx = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225])
    ])
    model_transfer.eval()
    lbls = get_labels()
    img = tx(img)
    img = torch.unsqueeze(img, 0)
    output = model_transfer(img)
    ind = torch.argmax(output, dim=1)
    return lbls[ind]
    