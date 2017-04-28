<p><strong>DOCUMENT UNDER EDITION</strong></p>
<h1 id="wako-record-improvement-probability">WAKO Record Improvement Probability</h1>
<h2 id="intro">Intro</h2>
<p>The WAKO city dataset is composed of 6497 electronic medical records (EMR). A record is a pair aaaa\((a,c)\), where \(a\) is a health assessment (<a href="https://www.evernote.com/shard/s25/nl/2147483647/fd37d473-bea9-41d5-92ec-927ab1f6b6b4/">assessment variables</a>) and \(c\) is a care plan. The assessment is a set of health variables measured by experts and doctors and the care plan is a sequence of health services \(s_{q}\).<br>
A record is also associated to a care level \(CL \in \{12,13,21,22,23,24,25\}\) and to an improvement label \(IL \in \{improve, maintain, decline\}\).</p>
<h2 id="task">Task</h2>
<p>We aim at learning a model to predict the improvement/decline of a new record. We learn the joint latent representation of care plans and health assessment variables and from this representation we infer the improvement/decline health outcome.</p>
<h2 id="preprocessing">Preprocessing</h2>
<p>To reduce ambiguities, we categorized all variables and created a dictionary \(D\). Its vocabulary is composed of 2064 words, which correspond to the possible observation values appearing in assessments and care plans. For categorizing the data:</p>
<ul>
<li>We considered empty observations (lack of data) variable dependent. Thus, we assigned a token per variable to represents them.</li>
<li>We considered the cross-modality of the numerical observations –the same observed value have different meaning across variables. To preserve meaning we assign a unique category variable observation.</li>
</ul>
<p>Our model is a forward neural network which takes vectorial inputs the concatenation \(i= a \oplus c\) where \(a\) represents an assessment and \(c\) a care plan, sampled from a dictionary \(D\). The model produces a binary classification output, e.i. \(improve\) or \(decline\). We don’t use \(maintain\) labeled records.</p>
<h4 id="assessment-and-careplan-representations">Assessment and careplan representations</h4>
<p>The assessment is a concatenation<br>
\[a = a_{expert} \oplus a_{doctor} \oplus a_{diseases} \oplus c_{services}\]<br>
where:</p>
<ul>
<li>\(a_{expert}\) contains a vector of 79 health variables provided by experts.</li>
<li>\(a_{doctor}\) contains a vector of 94 health variables measured by an independent medical doctor.</li>
<li>\(a_{diseases}\) contains a binary vector indicating the presence of certain diseases categorized by ICD10 standard. Per record, we can have up to 15 diseases. We have represented the information as a global vector \(a_{diseases}\) of 596 dimensions. Each dimension corresponds to a ICD10 disease of our dataset.  Each variable \(v_d \in a_{diseases}^{1,\dots,596}\) represents the presence or absence of the disease \(d\).</li>
<li>\(c\) contains the care plan services represented as a binary vector  of 276 dimensions, where 276 are all possible services in our dataset. A service is acctivated \(c_{services}^{i}=1\) when appears in the training example.</li>
</ul>
<p>All values part of the input are categorized following the rules mentioned in section <a href="#preprocessing">Preprocessing</a>.</p>
<h3 id="target">Target</h3>
<p>The model target is a one-hot vector \(Y = &lt;decline, improve&gt;\), where \(Y = &lt;1,0&gt;\) indicates the record’s careplan declines patient’s care level.</p>
<h3 id="embedding-layers">Embedding layers</h3>
<p>First, our model embeds the independent inputs in to a higher dimensional space. For doing, we learn two embedding infunctions \(l_a\),\(l_c\) for the inputs \(a\), \(c\).<br>
An embedding function \(l\) maps each variable observation to a feature vector of higher dimensionality by a look up table operation for which the parameters are learnt. For each word observation \(s \in D\), an embedding function \(l(·,W)\) produces a feature vector \(d^k\), where \(W \in \mathbb{R}^{k \times |D|}\) is learnt, \(k\) is a hyper-parameter and \(|D|\) is the dictionary length.</p>
<h3 id="encoder-layer">Encoder layer</h3>
<p>Let \(a' \in \mathbb{R}^{|a| \times k}\) and \(c' \in \mathbb{R}^{|c| \times k}\) be the output of the embedding layer of our model. For extracting the key information embedded in \(a'\) and \(c'\) we reduce the dimensionality by learning two affined transformations \(f_a: \mathbb{R}^{|a| \times k} \to \mathbb{R}^{100}\), \(f_c: \mathbb{R}^{|c| \times k} \to \mathbb{R}^{100}\) with ReLU activations for \(a'\), \(c'\) respectively.</p>
<h3 id="joint-representation">Joint representation</h3>
<p>We guarantee that both assessment and care plan are used for inferring the patient’s health outcome. Otherwise, the model could learn to dismiss the information on the care plan and predicting outcomes using only the assessment information. Such a prediction would be wrong since it would be expressing that the patient can improve with “some” care plan instead of the specific one we are providing as input. The opposite situation is also valid. To generate such a guarantee, we fuse the vector representations corresponding to an assessment \(f_a(a') = x_A \in \mathbb{R}^d\) and a care plan \(f_c(c') = x_C \in \mathbb{R}^d\) into the joint feature vector \(x_r \in  \mathbb{R}^n\). To obtain the \(x_r\) we learn the parameters \(W \in \mathbb{R}^{d \times d}\) of a bilinear transformation:<br>
\[x_r^i = f_a W^i f_c^T\]</p>
<h2 id="experiments">Experiments</h2>
<p>We aim at predicting the improvement label of a care plan implementation. We set up the task as a binary classification problem where we build a single algorithm to handle all care levels at once. We gathered the records corresponding to \(improve\) and \(decline\) labels.</p>
<p>For easing the understanding of sections below, we describe the experiment making reference to both models structure with the term <em>model</em>.</p>
<h3 id="loss">Loss</h3>
<p>We used as loss function the cross entropy:<br>
\[Y_i * log(g(i))\]<br>
where is the concatenation of assessment and careplan, e.i. \(i = a \oplus c\).</p>
<h3 id="train--validation">Train / Validation</h3>
<p>Each record \((a,c)\) is associated with an improvement label \(IL\), where \(a\) is a vector of the assessment variables \(&lt;a_{1},...,a_{769}&gt;\),  \(c\) is a set of services.</p>
<p>In our dataset, there are 36 pairs that shares the values of expert variables \(a_{1}\dots a_{79}\). All these records (in total 72) were removed from the set before constructing the dataset.</p>
<h4 id="experiments-1">Experiments</h4>
<p>We performeed two experiments, \(g\) and \(h\) are the models$ are different model, architecture are variations of the model architecture defined in <a href="#method">Method</a>. The difference between \(g\) and \(h\) relies in the input data. The model \(g\) takes all health variables in an assessment as input. Model \(h\) takes only the assessment variables measured by an expert.</p>
<p>We split the dataset in two sets:</p>
<ol>
<li>\(D_{nursing\_care}\): records with \(CL \in \{21,22,23,24,25\}\)</li>
<li>\(D_{linchpin\_support}\): records with \(CL \in \{12,13\}\)</li>
</ol>
<p>In Table 1, we depict the number of examples for each experiment arranged by the patient’s outcome.</p>
<table>
<thead>
<tr>
<th>Dataset</th>
<th>\(improve\)</th>
<th>\(decline\)</th>
</tr>
</thead>
<tbody>
<tr>
<td>\(D_{nursing\_care}\)</td>
<td>794</td>
<td>1350</td>
</tr>
<tr>
<td>\(D_{linchpin\_support}\)</td>
<td>96</td>
<td>227</td>
</tr>
</tbody>
</table>
<p><em>Table 1</em>: information and results obtained for each dataset.</p>
<p>The class <strong>RecordImprovementProbability</strong>, provides methods for training and inferring these models. This class is used by the notebook <em>run.ipynb</em>, where we train and evaluate this architecture using the data of Wako city.</p>
<h4 id="evaluation">Evaluation</h4>
<p>We measure the Accuracy of our method at classifying the improvement or decline of an (assessment, careplan) input pair. For each experiment we perform cross validation with 10 folds. Each fold was trained during 300 epochs.</p>
<h5 id="data-balancing">Data balancing</h5>
<p>Given that our experiment’s data appears unbalanced (see Table 1), at each fold we reduce the model bias sub-sampling. We <strong>down-sampled</strong> the decline examples to match the number of improve examples, fininshing with a 50/50% split.</p>
<h5 id="trainvalidation-sets">Train/Validation sets</h5>
<p>We construct the validation \((VAL)\) set by extracting 20% of examples from the balanced sets. For validation, the random chance of the model’s Accuracy is 0.5.</p>
<h5 id="results">Results</h5>
<p>In Table 2, we show the number of training/validation examples at each fold and the Mean Max Accuracy the experiments $D_{nursing_care} and \(D_{linchpin\_support}\).</p>
<table>
<thead>
<tr>
<th>Dataset</th>
<th>Folds</th>
<th>Training examples</th>
<th>Validation examples</th>
<th>\(g\) Mean Max Accuracy (Std)</th>
<th>\(h\) Mean Max Accuracy (Std)</th>
</tr>
</thead>
<tbody>
<tr>
<td>\(D_{nursing\_care}\)</td>
<td>10</td>
<td>1272</td>
<td>316</td>
<td>0.698 (0.015)</td>
<td>0.677 (0.017)</td>
</tr>
<tr>
<td>\(D_{linchpin\_support}\)</td>
<td>10</td>
<td>154</td>
<td>38</td>
<td>0.621 (0.049)</td>
<td>0.608 (0.048)</td>
</tr>
</tbody>
</table>
<p><em>Table 2</em>: information and results obtained for each dataset.</p>
<p>In <em>Figure 1</em> we show the confusion matrix of evaluating \(g\) on \(D_{nursing\_care}\) for a single fold.<br>
<img src="https://i.imgur.com/DqjFXX7.png" alt="Confusion matrix in a single fold"><br>
![Alt text](./Screen Shot 2017-04-01 at 18.54.16.png)</p>
<p><em>Figure 2</em>: confusion matrix of 1st fold done of model \(g\) trained with \(D_{nursing\_care}\).<br>
Available at: <a href="https://plot.ly/~guido.cs.stanford.edu/5350/care-levels-21-22-23-24-25-fold-9/">https://plot.ly/~guido.cs.stanford.edu/5350/care-levels-21-22-23-24-25-fold-9/</a></p>
<h4 id="paths-to-the-models-binary-files-in-our-server">Paths to the models’ binary files in our server</h4>
<p>The folds of \(g\) can be found in panda2 at: <br> /workspace/data/ai<br>
<em>core/experiments_results/WAKO_RecordImprovementProbability/v0/*</em><br>
The folds of \(h\) can be found in panda3 at:<br> /workspace/data/ai_core/experiments_results/WAKO_RecordImprovementProbability/v1/*</p>
<h4 id="production-usage">Production Usage</h4>
<p>Model \(g\) is used in production for estimating the record’s improvement and decline probability of each of the care plans suggested (\(c_{top_1}\), \(c_{top_2}\) and \(c_{fusion}\)) combined with the record’s assessment.<br>
For production, we trained 10 \(g\) models using different balanced subsets of \(D_{nursing\_care}\) and 10 others using different balanced subsets of \(D_{linchpin\_support}\).<br>
Given a new record \(r_{new} = (a_{new},c_{new})\),  we compose \[r_{i} = a_{new} \oplus c_{i},\  i \in \{top_1, top_2, fusion\}\]<br>
with the format described in section <a href="#input-format">Input Format</a>.<br>
Then, we pass each \(r_i\) through all the \(g\) models of the corresponding care level and display in the UI the mean improvement and decline probability for each \(c_i\).</p>
<h3 id="implementation">Implementation</h3>
<p>We use tensorflow with tflearn framework for defining, training and evaluating the model architecture. In our code, the model architecture takes as input the placeholder \(vars\) filled with a matrix of dim <em>d x 1045</em> for model \(g\) (<em>d x 355</em> for model \(h\)) where each row contains the variable values of an assessment \(a\) concatenated with the values of \(c_{services}\).<br>
We defined the target placeholder \(Y_{ph}\) as a matrix of size <em>dx2</em>.</p>
<h4 id="how-to-train-your-own-model">How to train your own model?</h4>
<ol>
<li>Open notebook /workspace/data/ai_core/WAKO_RecordImprovementProbability/vX/run.ipynb (X in [0,1])</li>
<li>Configure type of training to do (explained in the notebook).</li>
<li>Save changes on notebook</li>
<li>Go to terminal and run:</li>
</ol>
<pre class=" language-bash"><code class="prism  language-bash"><span class="token function">cd</span> /workspace/data/ai_core/WAKO_RecordImprovementProbability/v0

<span class="token comment" spellcheck="true">#generate .py</span>
jupyer nbconvert --to python run.ipynb

<span class="token comment" spellcheck="true">#train model Record Improvement Probability v0</span>
python run.py
</code></pre>
<h5 id="source-code">Source Code</h5>
<p>For model \(g\)<br>
<em>Model estimator</em>: /workspace/data/ai_core/WAKO_RecordImprovementProbability/v0/estimator.py<br>
<em>Training and evaluation</em>: /workspace/data/ai_core/WAKO_RecordImprovementProbability/v0/run.ipynb</p>
<p>For model \(h\):<br>
<em>Model estimator</em>: /workspace/data/ai_core/WAKO_RecordImprovementProbability/v1/estimator.py<br>
<em>Training and evaluation</em>: /workspace/data/ai_core/WAKO_RecordImprovementProbability/v1/run.ipynb</p>
