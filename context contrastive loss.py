from torch.utils.data import Dataset
import json
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns;

sns.set()
import torch
from argparse import ArgumentParser
import copy
import torch.nn.functional as F
import random
from PIL import Image
import tensorboardX

parser = ArgumentParser(description='context contrastive loss')
parser.add_argument('--batch_size', type=int, default=16)
parser.add_argument('--contrastive_loss_margin', type=float, default=0.8)
parser.add_argument('--learning_rate', type=float, default=1e-5)
parser.add_argument('--num_epochs', type=int, default=100)
parser.add_argument('--weight_decay', type=float, default=1e-5)
parser.add_argument('--mode', type=str, default='train')
parser.add_argument('--device', type=str, default='cuda')
parser.add_argument('--img_size', type=int, default=200)
parser.add_argument('--no_of_samples_per_class', type=int, default=100)
parser.add_argument('--no_of_classes', type=int, default=50)
parser.add_argument('--projector_img_size', type=int, default=32)
parser.add_argument('--number_of_positive_pairs_per_class', type=int, default=5)

args, unknown = parser.parse_known_args()

DEVICE = args.device
if not torch.cuda.is_available():
    DEVICE = 'cpu'


# previous trained encoder model
class Encoder(nn.Module):

    def __init__(self):
        super(Encoder, self).__init__()

        # conv and fc works as encoder
        self.conv = nn.Sequential(nn.Conv2d(in_channels=3, out_channels=32, kernel_size=5),
                                  nn.ReLU(),
                                  nn.MaxPool2d(kernel_size=2, stride=2),
                                  nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5),
                                  nn.ReLU(),
                                  nn.MaxPool2d(kernel_size=2, stride=2),
                                  nn.Conv2d(in_channels=64, out_channels=128, kernel_size=5),
                                  nn.ReLU(),
                                  nn.MaxPool2d(kernel_size=2, stride=2)
                                  )

        # output 128, 21, 21
        self.fc = nn.Sequential(nn.Linear(128 * 21 * 21, 1024),
                                nn.ReLU(),
                                nn.Linear(1024, 1024),
                                nn.ReLU(),
                                nn.Linear(1024, 128)
                                )

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size()[0], -1)
        out_x = self.fc(x)
        l2_length = torch.norm(out_x.detach(), p=2, dim=1, keepdim=True)
        z = out_x / l2_length
        return z


class ContextDavis(Dataset):

    def __init__(self, json_data, memmap):
        self.memmap = memmap
        self.json_data = json_data
        self.shape = self.json_data["shape"]
        self.objects = self.json_data["objects"]
        self.contexts = self.json_data["contexts"]
        self.keys_of_images = list(self.contexts.keys())
        self.keys_of_images.sort()
        self.start_img_index = int(self.keys_of_images[0])
        self.length = len(self.keys_of_images)

    def __getitem__(self, index):
        correct_index = self.start_img_index + index

        contexts_per_image = self.contexts[str(correct_index)]

        img_objects = []
        img_classes = []

        for context_image in contexts_per_image:
            img_object = self.objects[str(context_image)]
            img_object_width = img_object['width']
            img_object_height = img_object['height']
            image_ = self.memmap[context_image, :, :img_object_width, :img_object_height].astype(np.float32)
            class_ = img_object['class']

            image_tensor_ext = torch.from_numpy(image_).unsqueeze(dim=0)
            image_tensor = F.interpolate(image_tensor_ext, size=(args.img_size, args.img_size))
            image_tensor = image_tensor.squeeze(dim=1)

            img_objects.append(image_tensor)
            img_classes.append(class_)

        return img_objects, img_classes

    def __len__(self):
        return self.length


class Context_Model(nn.Module):

    def __init__(self):
        super(Context_Model, self).__init__()

        self.fc = nn.Sequential(nn.Linear(256, 256),
                                nn.ReLU(),
                                nn.Linear(256, 128)
                                )

    def forward(self, concat_zs):
        pair_img_tensor_1 = concat_zs[0]
        pair_img_tensor_2 = concat_zs[1]
        context_aggregated_z1 = self.fc(pair_img_tensor_1)
        context_aggregated_z2 = self.fc(pair_img_tensor_2)
        return context_aggregated_z1, context_aggregated_z2


class ContrastiveLoss(nn.Module):

    def __init__(self, margin):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output, target):
        eq_distance = F.pairwise_distance(output[0], output[1])
        loss = 0.5 * (1 - target) * torch.pow(eq_distance, 2) + \
               0.5 * target * torch.pow(torch.clamp(self.margin - eq_distance, min=0.00), 2)

        return loss.mean()


def collate_for_same_order(batch):
    data = [item[0] for item in batch]
    target = [item[1] for item in batch]
    return [data, target]


def draw_loss_plot(training_losses, validation_losses, epochs):
    plt.plot(epochs, training_losses, label="Train")
    plt.plot(epochs, validation_losses, label="eval")
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.legend()
    plt.show()


def transform_image_for_projector(img_tensor):
    x_np = img_tensor.to('cpu').data.numpy()
    x_np = x_np.swapaxes(0, 1)
    x_np = x_np.swapaxes(1, 2)
    # H, W, C
    img = Image.fromarray(x_np.astype(np.uint8), mode='RGB')

    img = img.resize((args.projector_img_size, args.projector_img_size), Image.ANTIALIAS)
    img = np.array(img).astype(np.float)

    img = img.swapaxes(2, 1)
    img = img.swapaxes(1, 0)
    img /= 255

    return img


folder_path = '../Memmaps_Jsons/'
with open(folder_path + 'train_davis.json') as json_file:
    train_davis_json = json.load(json_file)
with open(folder_path + 'test_davis.json') as json_file:
    test_davis_json = json.load(json_file)

train_shape = train_davis_json["shape"]
train_memmap_path = folder_path + 'train_davis.mmap'
train_davis_memmap = np.memmap(train_memmap_path, dtype='uint8', mode='r', shape=tuple(train_shape))

test_shape = test_davis_json["shape"]
test_memmap_path = folder_path + 'test_davis.mmap'
test_davis_memmap = np.memmap(test_memmap_path, dtype='uint8', mode='r', shape=tuple(test_shape))

train_davis_dataset = ContextDavis(train_davis_json, train_davis_memmap)
train_davis_dataloader = torch.utils.data.DataLoader(train_davis_dataset, batch_size=args.batch_size, shuffle=True,
                                                     collate_fn=collate_for_same_order)

test_davis_dataset = ContextDavis(test_davis_json, test_davis_memmap)
test_davis_dataloader = torch.utils.data.DataLoader(test_davis_dataset, batch_size=args.batch_size, shuffle=False,
                                                    collate_fn=collate_for_same_order)

# load previously trained encoder model
model_path = './encoder_1.pth'
encoder_model = Encoder()
# encoder_model.load_state_dict(torch.load(model_path, map_location='cpu'))
encoder_model.load_state_dict(torch.load(model_path))
encoder_model = encoder_model.to(DEVICE)
encoder_model.eval()
# torch.load(load_path, map_location=map_location)

tensorboard_writer = tensorboardX.SummaryWriter()
model_save_path = './context_2.pth'

context_model = Context_Model()
context_model = context_model.to(DEVICE)
context_model.train()

criterion = ContrastiveLoss(margin=args.contrastive_loss_margin)
optimizer = torch.optim.Adam(params=context_model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

epoches = []
train_losses = []
eval_losses = []

for epoch in range(1, args.num_epochs + 1):

    train_batch_losses = []
    eval_batch_losses = []
    epoches.append(epoch)
    stage = ''

    for dataloader in [train_davis_dataloader,test_davis_dataloader]:

        classes_dict = {}
        projector_labels = []
        projector_imgs = []
        projector_embeddings = []
        counter = 0

        for batch in dataloader:

            batch_img_objects, batch_classes = batch[0], batch[1]

            batch_z_vectors = []

            # getting embedding vectors for img_objects
            torch.set_grad_enabled(False)
            for img_objects in batch_img_objects:  # groups
                z_vectors = []
                for single_img_object in img_objects:
                    single_img_object = single_img_object.to(DEVICE)
                    z_vectors.append(encoder_model(single_img_object.squeeze(dim=1)))
                batch_z_vectors.append(z_vectors)

            # batch_z_vectors  [[a,b,c] [c,d] [e,f,g]]
            # batch_img_objects [[imga,imgb,imgc] [imgc,imgd] [..]]

            # getting avg vectors and labels for context classes
            batch_processed_z_vector_pairs = []
            batch_processed_class_labels = []
            batch_images_for_main_z = []

            for z_vectors, classes, images in zip(batch_z_vectors, batch_classes, batch_img_objects):
                Z_vector_pair = []
                classes_pair = []
                image_main_z = []
                for single_zs, class_, image in zip(z_vectors, classes, images):
                    main_z = single_zs
                    avg_z = torch.mean(torch.stack(z_vectors), dim=0)
                    Z_vector_pair.append(main_z)
                    Z_vector_pair.append(avg_z)
                    image_main_z.append(image)

                    classes_pair.append(class_)
                    temp_classes = copy.deepcopy(classes)
                    temp_classes.remove(class_)
                    avg_class_label = ''
                    for temp_clz in temp_classes:
                        if temp_clz not in avg_class_label:
                            avg_class_label = avg_class_label + '-' + temp_clz
                    context_class_label = class_ + '-in' + avg_class_label
                    classes_pair.append(context_class_label)

                batch_processed_z_vector_pairs.append(Z_vector_pair)
                batch_processed_class_labels.append(classes_pair)
                batch_images_for_main_z.append(image_main_z)

            # print('len of z pairs',len(batch_processed_z_vector_pairs))
            # print('len of labels',len(batch_processed_class_labels))
            # print('len of imgaes',len(batch_images_for_main_z))

            # batch_processed_z_vector_pairs  [[mz,az,mz,az,mz,az], [mz,az,mz,az,mz,az]]
            # batch_processed_class_labels [['person', 'person-in-horse', 'horse', 'horse-in-person-horse', 'horse', 'horse-in-person-horse'],[...]]
            # batch_images_for_main_z [[img1,img2,img3], [img4, img5,img6], ...]

            tensor_main_avg_z = torch.Tensor()
            tensor_main_avg_z = tensor_main_avg_z.to(DEVICE)
            tensor_main_img = torch.Tensor()
            tensor_main_img = tensor_main_img.to(DEVICE)
            context_aggregated_labels = []

            # getting context aggregated zs from model and the aggregated class lables for them
            for processed_z_vector_pairs, processed_class_label_pairs, image_main_z in zip(
                    batch_processed_z_vector_pairs,
                    batch_processed_class_labels, batch_images_for_main_z):
                for main_z_idx, avg_z_idx, img_idx in zip(range(0, len(processed_class_label_pairs), 2),
                                                          range(1, len(processed_class_label_pairs), 2),
                                                          range(len(image_main_z))):
                    # print(processed_class_label_pairs[main_z_idx])
                    context_aggregated_labels.append(processed_class_label_pairs[main_z_idx])  ##changed
                    main_z = processed_z_vector_pairs[main_z_idx]
                    avg_z = processed_z_vector_pairs[avg_z_idx]
                    main_img = image_main_z[img_idx]
                    main_z = main_z.to(DEVICE)
                    avg_z = avg_z.to(DEVICE)
                    main_img = main_img.to(DEVICE)
                    concat_main_avg = torch.cat((main_z, avg_z), dim=1)  # concat main_z and avg_z of single object
                    concat_main_avg = concat_main_avg.to(DEVICE)
                    tensor_main_avg_z = torch.cat((tensor_main_avg_z, concat_main_avg))  # batch of concat_main_avg
                    tensor_main_img = torch.cat((tensor_main_img, main_img))

            # print('shape of main avg z', tensor_main_avg_z.shape)
            # print('shape of main imgs', tensor_main_img.shape)

            # create context_labels with context_zs
            class_labels_with_context = {}
            images_for_main_z = {}
            for class_idx, class_label in enumerate(context_aggregated_labels):
                if class_label not in list(class_labels_with_context.keys()):
                    class_labels_with_context[class_label] = [tensor_main_avg_z[class_idx]]
                    images_for_main_z[class_label] = [tensor_main_img[class_idx]]
                else:
                    current_z_list = class_labels_with_context.get(class_label)
                    current_z_list.append(tensor_main_avg_z[class_idx])
                    class_labels_with_context[class_label] = current_z_list
                    current_img_list = images_for_main_z.get(class_label)
                    current_img_list.append(tensor_main_img[class_idx])
                    images_for_main_z[class_label] = current_img_list

            # print(class_labels_with_context)
            # print(images_for_main_z)

            # create similiar, disimilar pairs
            context_z_pair_img_1 = []
            context_z_pair_img_2 = []
            pair_img1 = []
            pair_img2 = []
            targets = []
            class_lables = []
            total_positive_pairs = 0

            # similiar pairs
            for class_label in list(class_labels_with_context.keys()):
                z_vectors = class_labels_with_context.get(class_label)
                img_for_label = images_for_main_z.get(class_label)
                counter_per_class = 0
                if len(z_vectors) > 2:
                    while counter_per_class < args.number_of_positive_pairs_per_class:
                        selected_two_idxes = random.choices(range(len(z_vectors)), k=2)

                        context_z_pair_img_1.append(z_vectors[selected_two_idxes[0]])
                        context_z_pair_img_2.append(z_vectors[selected_two_idxes[1]])

                        pair_img1.append(img_for_label[selected_two_idxes[0]])
                        pair_img2.append(img_for_label[selected_two_idxes[0]])

                        targets.append(0)
                        class_lables.append([class_label, class_label])
                        counter_per_class += 1
                        total_positive_pairs += 1

            # dissimilar pairs
            counter_dissimilar_pair = 0
            while counter_dissimilar_pair < total_positive_pairs:
                pair_keys = np.random.choice(list(class_labels_with_context.keys()), size=2, replace=False)

                context_z_pair_1_idx = random.choices(range(len(class_labels_with_context.get(pair_keys[0]))), k=1)
                context_z_pair_2_idx = random.choices(range(len(class_labels_with_context.get(pair_keys[1]))), k=1)

                context_z_pairs_1_for_key = class_labels_with_context.get(pair_keys[0])
                context_z_pairs_2_for_key = class_labels_with_context.get(pair_keys[1])

                context_z_pair_img_1.append(context_z_pairs_1_for_key[context_z_pair_1_idx[0]])
                context_z_pair_img_2.append(context_z_pairs_2_for_key[context_z_pair_2_idx[0]])

                pair_img1.append(images_for_main_z.get(pair_keys[0])[context_z_pair_1_idx[0]])
                pair_img2.append(images_for_main_z.get(pair_keys[1])[context_z_pair_2_idx[0]])

                class_lables.append(pair_keys.tolist())
                targets.append(1)
                counter_dissimilar_pair += 1

            if dataloader == train_davis_dataloader:
                context_model.train()
                torch.set_grad_enabled(True)
                stage = 'train'
            else:
                context_model.eval()
                torch.set_grad_enabled(False)
                stage = 'eval'

            # pass to model
            context_model = context_model.to(DEVICE)

            tensor_context_z_pair_img_1 = torch.stack(context_z_pair_img_1, dim=0)
            tensor_context_z_pair_img_2 = torch.stack(context_z_pair_img_2, dim=0)

            tensor_context_z_pair_img_1 = tensor_context_z_pair_img_1.to(DEVICE)
            tensor_context_z_pair_img_2 = tensor_context_z_pair_img_2.to(DEVICE)

            tensor_context_z_pair = [tensor_context_z_pair_img_1, tensor_context_z_pair_img_2]

            context_aggregated_z1, context_aggregated_z2 = context_model(tensor_context_z_pair)

            out = [context_aggregated_z1, context_aggregated_z2]

            tensor_target = torch.FloatTensor(targets)
            tensor_target = tensor_target.to(DEVICE)

            loss = criterion(out, tensor_target)

            if dataloader == train_davis_dataloader:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                train_batch_losses.append(loss.item())
            else:
                eval_batch_losses.append(loss.item())

            # if args.num_epochs > 12:
            # args.learning_rate = 1e-5

            # adding to tensorboard projector

            context_model.eval()
            torch.set_grad_enabled(False)

            # pass data again to model to embeddings for the projector

            context_aggregated_z1, context_aggregated_z2 = context_model(tensor_context_z_pair)

            for pair_idx, class_pairs in enumerate(class_lables):
                for class_idx, single_class in enumerate(class_pairs):
                    if len(classes_dict.keys()) < args.no_of_classes:
                        if single_class not in classes_dict.keys():
                            classes_dict[single_class] = 1
                            projector_labels.append(single_class)
                            if class_idx == 0:
                                projector_embeddings.append(context_aggregated_z1.cpu()[pair_idx])
                                projector_imgs.append(transform_image_for_projector(pair_img1[pair_idx]))
                            else:
                                projector_embeddings.append(context_aggregated_z2.cpu()[pair_idx])
                                projector_imgs.append(transform_image_for_projector(pair_img2[pair_idx]))
                        else:
                            current_count = classes_dict.get(single_class)
                            if current_count <= args.no_of_samples_per_class:
                                classes_dict[single_class] = current_count + 1
                                projector_labels.append(single_class)
                                if class_idx == 0:
                                    projector_embeddings.append(context_aggregated_z1.cpu()[pair_idx])
                                    projector_imgs.append(transform_image_for_projector(pair_img1[pair_idx]))
                                else:
                                    projector_embeddings.append(context_aggregated_z2.cpu()[pair_idx])
                                    projector_imgs.append(transform_image_for_projector(pair_img2[pair_idx]))

        # at end of the epoch
        if dataloader == train_davis_dataloader:
            print('Epoch : ', epoch, 'Stage : ', stage, 'Loss : ', np.mean(train_batch_losses))
            train_losses.append(np.mean(train_batch_losses))
            tensorboard_writer.add_scalars(tag_scalar_dict={'Train': np.mean(train_batch_losses)}, global_step=epoch,
                                           main_tag='Loss')
            torch.save(context_model.state_dict(), model_save_path)

        else:
            print('Epoch : ', epoch, 'Stage : ', stage, 'Loss : ', np.mean(eval_batch_losses))
            eval_losses.append(np.mean(eval_batch_losses))
            tensorboard_writer.add_scalars(tag_scalar_dict={'Eval': np.mean(eval_batch_losses)}, global_step=epoch,
                                           main_tag='Loss')

        tensorboard_writer.add_embedding(
            mat=torch.FloatTensor(np.stack(projector_embeddings)),
            label_img=torch.FloatTensor(np.stack(projector_imgs)),
            metadata=projector_labels,
            global_step=epoch, tag=f'{stage}_emb_{epoch}')
        tensorboard_writer.flush()

#draw_loss_plot(train_losses, eval_losses, epoches)





