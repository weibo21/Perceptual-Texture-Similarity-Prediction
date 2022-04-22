class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        vgg = models.vgg19(pretrained=True).features
        self.conv1_1 = vgg[0:2]
        self.conv1_2 = vgg[2:4]
        self.conv2_1 = vgg[4:7]
        self.conv2_2 = vgg[7:9]
        self.conv3_1 = vgg[9:12]
        self.conv3_2 = vgg[12:14]
        self.conv3_3 = vgg[14:16]
        self.conv3_4 = vgg[16:18]
        self.conv4_1 = vgg[18:21]
        self.conv4_2 = vgg[21:23]
        self.conv4_3 = vgg[23:25]
        self.conv4_4 = vgg[25:27]
        self.conv5_1 = vgg[27:30]
        self.conv5_2 = vgg[30:32]
        self.conv5_3 = vgg[32:34]
        self.conv5_4 = vgg[34:36]

        self.out = nn.Sequential(
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, 1)
            # nn.Sigmoid()
        )

    def calculateSimilarity(self, y1, y2, l):
        y1_n = torch.nn.functional.normalize(y1, p=2, dim=1)
        y2_n = torch.nn.functional.normalize(y2, p=2, dim=1)
        sim_pre = torch.mean(torch.sum(y1_n * y2_n, dim=1), dim=(1, 2))
        return sim_pre.unsqueeze(dim=1)

    def forward(self, x1, cx1, x2, cx2):
        x = torch.cat((x1, cx1, x2, cx2))
        conv1_1 = self.conv1_1(x)
        conv1_2 = self.conv1_2(conv1_1)
        conv2_1 = self.conv2_1(conv1_2)
        conv2_2 = self.conv2_2(conv2_1)
        conv3_1 = self.conv3_1(conv2_2)
        conv3_2 = self.conv3_2(conv3_1)
        conv3_3 = self.conv3_3(conv3_2)
        conv3_4 = self.conv3_4(conv3_3)
        conv4_1 = self.conv4_1(conv3_4)
        conv4_2 = self.conv4_2(conv4_1)
        conv4_3 = self.conv4_3(conv4_2)
        conv4_4 = self.conv4_4(conv4_3)
        conv5_1 = self.conv5_1(conv4_4)
        conv5_2 = self.conv5_2(conv5_1)
        conv5_3 = self.conv5_3(conv5_2)
        conv5_4 = self.conv5_4(conv5_3)

        features = [conv1_1, conv1_2, conv2_1, conv2_2, conv3_1, conv3_2, conv3_3, conv3_4,
                    conv4_1, conv4_2, conv4_3, conv4_4, conv5_1, conv5_2, conv5_3, conv5_4]

        isimilarities = []
        csimilarities = []

        for l, f in enumerate(features):
            b1, b2 = torch.chunk(f, 2, 0)
            if1, cf1 = torch.chunk(b1, 2, 0)
            if2, cf2 = torch.chunk(b2, 2, 0)
            isimilarities.append(self.calculateSimilarity(if1, if2, l))
            csimilarities.append(self.calculateSimilarity(cf1, cf2, l))

        isimilarities = torch.stack(isimilarities, dim=1)
        csimilarities = torch.stack(csimilarities, dim=1)

        similarities = torch.cat((isimilarities, csimilarities), dim=1).squeeze()

        return self.out(similarities)
