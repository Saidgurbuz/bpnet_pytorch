import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions.multinomial import Multinomial


def multinomial_nll(true_counts, logits):
	"""
	Compute the multinomial negative log-likelihood along the sequence (axis=1)
	and sum the values across all each channels

	Args:
		true_counts: observed count values (batch, seqlen, channels)
		logits: predicted logit values (batch, seqlen, channels)
	"""
	logits_perm = logits.permute(0, 2, 1)
	true_counts_perm = true_counts.permute(0, 2, 1)

	counts_per_example = true_counts_perm.sum(dim=-1)


	multinomial_dist = Multinomial(counts_per_example, logits=logits_perm)

	batch_size = float(true_counts.size(0))

	neg_log_likelihood = -multinomial_dist.log_prob(true_counts_perm)

	loss = neg_log_likelihood.sum() / batch_size

	return loss

class BPNet(nn.Module):
	def __init__(self, num_tasks):
		super(BPNet, self).__init__()
		self.num_tasks = num_tasks

		# Body (Convolutional Layers)
		self.conv1 = nn.Conv1d(4, 64, kernel_size=25, padding='same')
		self.relu = nn.ReLU()
		self.conv_layers = nn.ModuleList([
			nn.Conv1d(64, 64, kernel_size=3, padding='same', dilation=2**i)
			for i in range(1, 10)
		])

		# Heads (Output Layers)
		self.profile_shape_heads = nn.ModuleList([
			nn.Sequential(
				nn.ConvTranspose2d(64, 2, kernel_size=(25, 1), padding='same'),
				nn.Flatten(),
			)
			for _ in range(num_tasks)
		])
		
		self.total_counts_heads = nn.ModuleList([
			nn.Sequential(
				nn.AdaptiveAvgPool1d(1),
				nn.Flatten(),
				nn.Linear(64, 2)
			)
			for _ in range(num_tasks)
		])

	def forward(self, x):
		# Body
		x = self.conv1(x)
		x = self.relu(x)
		for conv_layer in self.conv_layers:
			x = self.relu(conv_layer(x))

		# Profile Shape and Total Counts Heads
		profile_shape_outputs = [head(x.unsqueeze(2)) for head in self.profile_shape_heads]
		total_counts_outputs = [head(x) for head in self.total_counts_heads]

		return profile_shape_outputs + total_counts_outputs

num_tasks = 4  # We are using 4 TFs
model = BPNet(num_tasks)

learning_rate = 0.004
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

loss_functions = [multinomial_nll, F.mse_loss] * num_tasks
loss_weights = [1, 10] * num_tasks

def custom_loss(outputs, targets):
	total_loss = 0.0
	for i, loss_fn in enumerate(loss_functions):
		total_loss += loss_weights[i] * loss_fn(outputs[i], targets[i])
	return total_loss
    

# Early stopping parameters
patience = 5
best_val_loss = float('inf')
counter = 0

# Training loop
num_epochs = 100
for epoch in range(num_epochs):
	model.train()
	train_loss = 0.0

	for data in train_loader:
		inputs, targets = data
		optimizer.zero_grad()
		outputs = model(inputs)
		
		loss = custom_loss(outputs, targets)
		
		loss.backward()
		optimizer.step()
		
		train_loss += loss.item()
		
	# validation loss
	model.eval()
	val_loss = 0.0
		
	with torch.no_grad():
		for data in val_loader:
			inputs, targets = data
			outputs = model(inputs)
			val_loss += custom_loss(outputs, targets).item()
		
	# training and validation loss for each epoch
	print(f'Epoch [{epoch + 1}/{num_epochs}] - Train Loss: {train_loss / len(train_loader)} - Validation Loss: {val_loss / len(val_loader)}')
		
	# for early stopping
	if val_loss < best_val_loss:
		best_val_loss = val_loss
		counter = 0
	else:
		counter += 1
		if counter >= patience:
			print(f'Early stopping after {epoch+1} epochs without improvement.')
			break

