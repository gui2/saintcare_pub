<p><strong>DOCUMENT UNDER EDITION</strong></p>
<h1 id="wako-record-improvement-probability">WAKO Record Improvement Probability</h1>
<h2 id="intro">Intro</h2>
<p>The WAKO city dataset is composed of 6497 electronic medical records (EMR). A record is a pair \((a,c)\), where \(a\) is a health assessment (<a href="https://www.evernote.com/shard/s25/nl/2147483647/fd37d473-bea9-41d5-92ec-927ab1f6b6b4/">assessment variables</a>) and \(c\) is a care plan. The assessment is a set of health variables measured by experts and doctors and the care plan is a sequence of health services \(s_{q}\).<br>
A record is also associated to a care level \(CL \in \{12,13,21,22,23,24,25\}\) and to an improvement label \(IL \in \{improve, maintain, decline\}\).</p>
<h2 id="task">Task</h2>
<p>Building a model capable of predicting the improvement label of a new given record by learning the joint latent representation of care plans and health assessment variables.</p>
<h2 id="how">How</h2>
<p>We learn a feed forward neural network which takes vectorial inputs \(a\) (assessment) and \(c\) (care plan), sampled from a dictionary \(D\), and produces a binary classification output between the improvement labels \(improve\) and \(decline\). Records with \(IL = maintain\) are not considered for this experiment.</p>
<h2 id="preprocessing">Preprocessing</h2>
<p>Our input data is multimodal and incomplete. To reduce ambiguities, we categorized all variables and created a dictionary \(D\). Its vocabulary is composed of 2064 words, which correspond to the possible observation values appearing in assessments and care plans. For categorizing the data:</p>
<ul>
<li>We considered empty observations (lack of data) variable dependent. Thus, we assigned a token per variable to represents them.</li>
<li>We considered the cross-modality of the numerical observations –the same observed value have different meaning across variables. To preserve meaning we assign a unique category variable observation.</li>
</ul>
<h2 id="method">Method</h2>
<p>First, our model embeds the independent inputs in to a higher dimensional space. For doing, we learn two embedding functions \(l_a\),\(l_c\) for the inputs \(a\), \(c\).</p>
<h3 id="embedding-layer">Embedding layer</h3>
<p>An embedding function \(l\) maps each variable observation to a feature vector of higher dimensionality by a look up table operation for which the parameters are learnt. For each word observation \(s \in D\), an embedding function \(l(·,W)\) produces a feature vector \(d^k\), where \(W \in \mathbb{R}^{k \times |D|}\) is learnt, \(k\) is a hyper-parameter and \(|D|\) is the dictionary length.</p>
<h3 id="encoder-layer">Encoder layer</h3>
<p>Let \(a' \in \mathbb{R}^{|a| \times k}\) and \(c' \in \mathbb{R}^{|c| \times k}\) be the output of the embedding layer of our model. For extracting the key information embedded in \(a'\) and \(c'\) we reduce the dimensionality by learning two affined transformations \(f_a: \mathbb{R}^{|a| \times k} \to \mathbb{R}^{100}\), \(f_c: \mathbb{R}^{|c| \times k} \to \mathbb{R}^{100}\) with ReLU activations for \(a'\), \(c'\) respectively.</p>
<h3 id="joint-representation">Joint representation</h3>
<p>We need to guarantee that both assessment and care plan are used for inferring the patient’s health outcome (represented by the improvement label). Otherwise, the model could learn to dismiss the information on the care plan and predicting outcomes using only the assessment information. Such a prediction would be wrong since it would be expressing that the patient can improve with “some” care plan instead of the specific one we are providing as input. The opposite situation is also valid. To generate such a guarantee, we fuse the vector representations corresponding to an assessment \(f_a(a') = x_A \in \mathbb{R}^d\) and a care plan \(f_c(c') = x_C \in \mathbb{R}^d\) into the joint feature vector \(x_r \in  \mathbb{R}^n\). To obtain the \(x_r\) we learn the parameters \(W \in \mathbb{R}^{d \times d}\) of a bilinear transformation:<br>
\[x_r^i = x_A W^i x_C^T\]</p>
<h2 id="experiment">Experiment</h2>
<p>We aim at predicting the improvement label of a care plan implementation. We set up the task as a binary classification problem where we build a single algorithm to handle all care levels at once. We gathered the data corresponding to \(improve\) and \(decline\) labels. Meanwhile, the \(maintain\) data is not used at this stage.</p>
<p>We defined the models \(g\) and \(h\), both with the architecture defined <a href="#method">above</a>. The only difference between \(g\) and \(h\) in the input data. Model \(g\) considers all the health variables of assessment \(a\) while \(h\) considers only the assessment variables measured by an expert.</p>
<p>For easing the understanding of sections below, we describe the experiment making reference to both models structure with the term <em>model</em>.</p>
<h3 id="input-format">Input format</h3>
<p>The model input is the concatenation of \(a\) and \(c\). We defined an assessment as<br>
\[a = a_{expert} \oplus a_{doctor} \oplus a_{diseases}\]<br>
where \(\oplus\) is the concatenation operator and the subparts are:</p>
<ul>
<li>\(a_{expert}\):  the vector of 79 health variables provided by experts.</li>
<li>\(a_{doctor}\) the vector of 94 variables measured by an independent medical doctor.</li>
<li>\(a_{diseases}\): This indicates the presence of certain diseases categorized by ICD10 standard. Per record, we can have up to 15 diseases. We have represented the information as a global vector \(a_{diseases}\) of 596 dimensions. Each dimension corresponds to a ICD10 disease of our dataset.  Each variable \(v_d \in a_{diseases}^{1,\dots,596}\) represents the presence or absence of the disease \(d\).</li>
</ul>
<p>We represent a care plan as a vector \(c_{services}\) of 276 dimensions. Each dimension corresponds to a health service of our dataset and indicate the presence or absence of this service in the care plan.</p>
<p>All values part of the input are categorized following the rules mentioned in section <a href="#preprocessing">Preprocessing</a>.</p>
<h3 id="target-format">Target format</h3>
<p>The target of the model is a one-hot vector \(&lt;y_{decline}, y_{improve}&gt;\) of dimension 2, where  \(y_{decline}\) indicates the record’s careplan declines patient’s care level and \(y_{improve}\) indicates the record’s careplan improves patient’s care level. Note, that these two classes are mutually exclusive.</p>
<h3 id="train--validation-data-sets">Train / Validation data sets</h3>
<p>Each record \((a,c)\) is associated with an improvement label \(IL\), where \(a\) is a vector of assessment variables \(&lt;a_{1},...,a_{769}&gt;\),  \(c\) is a set of services and \(IL \in  \{improve, maintain, decline\}\) indicates if the care plan improves, maintains or declines the care level of the patient.</p>
<p>In our dataset, there are 36 pairs that shares the values of expert variables \(a_{1}\dots a_{79}\).  All these records (in total 72) were removed from the set before constructing the dataset.</p>
<p>After removing records with \(IL = maintain\), we split the dataset in two:</p>
<ol>
<li>\(D_{nursing\_care}\): records with \(CL \in \{21,22,23,24,25\}\)</li>
<li>\(D_{linchpin\_support}\): records with \(CL \in \{12,13\}\)</li>
</ol>
<p>Let \(POS\) the set of records with \(IL = improve\) and \(NEG\) the set of records with \(IL = decline\):</p>
<table>
<thead>
<tr>
<th>Dataset</th>
<th>\(POS\)</th>
<th>\(NEG\)</th>
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
<p>Because \(POS\) and \(NEG\) are unbalanced in both datasets, we <strong>down-sampled</strong> the decline examples to match the number of improve examples.</p>
<h3 id="implementation">Implementation</h3>
<p>We use tensorflow with tflearn framework for defining, training and evaluating the model architecture. In our code, the model architecture takes as input the placeholder \(vars\) filled with a matrix of dim <em>d x 1045</em> for model \(g\) (<em>d x 355</em> for model \(h\)) where each row contains the variable values of an assessment \(a\) concatenated with the values of \(c_{services}\).</p>
<p>We defined the target placeholder \(Y_{ph}\) as a matrix of size <em>dx2</em>.<br>
We used as loss function the cross entropy:<br>
\[Y_{i} * log(g(a_i))\]<br>
where \(Y_i=1\).<br>
As the metric, we used the accuracy.</p>
<p>The class <strong>RecordImprovementProbability</strong>, provides methods for training and inferring these models. This class is used by the notebook <em>run.ipynb</em>, where we train and evaluate this architecture using the data of Wako city.</p>
<h4 id="evaluation">Evaluation</h4>
<p>In Table 1 we show the training data and accuracy evaluation of the models trained with records of care levels  \(\{21, \dots ,25\}\) (\(D_{nursing\_care}\)) and with records of care levels \(\{12,13\}\) (\(D_{linchpin\_support}\)). We have trained 10 folds during 300 epochs, and measured accuracy using cross-validation. For cross-validation, we held out a balanced 50/50% subset of Positive and Negative examples. Positive examples are the records in \(POS\) and negatives examples are the records in \(NEG\). Because the dataset was <strong>down-sampled</strong> beforehand, \(POS\) and \(NEG\) contain the same count of examples.</p>
<p>We construct the validation \((VAL)\) set by extracting 20% of examples from \(POS\) and \(NEG\).  Because \(POS\)  and \(NEG\) are balanced, \(VAL\) is balanced. With such a balanced \(VAL\) set, the random chance of the model’s Accuracy in the validation set is 0.5.</p>
<table>
<thead>
<tr>
<th>Dataset</th>
<th>Folds</th>
<th>Training examples</th>
<th>Validation examples</th>
<th>\(g\) accuracy <br> <em>Mean (Std)</em></th>
<th>\(h\) accuracy <br> <em>Mean (Std)</em></th>
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
<p><em>Table 1</em>: information and results obtained for each dataset.</p>
<p>![Alt text](./Screen Shot 2017-04-01 at 18.51.04.png)<br>
<em>Figure 1</em>: training curves of 10 folds of model \(g\) trained with \(D_{nursing\_care}\)</p>
<p>![Alt text](./Screen Shot 2017-04-01 at 18.54.16.png)<br>
<em>Figure 2</em>: confusion matrix of 1st fold done of model \(g\) trained with \(D_{nursing\_care}\).<br>
Available at: <a href="https://plot.ly/~guido.cs.stanford.edu/5350/care-levels-21-22-23-24-25-fold-9/">https://plot.ly/~guido.cs.stanford.edu/5350/care-levels-21-22-23-24-25-fold-9/</a></p>
<h4 id="production-usage">Production Usage</h4>
<p>Model \(g\) is used in production for estimating the record’s improvement and decline probability of each of the care plans suggested (\(c_{top_1}\), \(c_{top_2}\) and \(c_{fusion}\)) combined with the record’s assessment.<br>
For production, we trained 10 \(g\) models using different balanced subsets of \(D_{nursing\_care}\) and 10 others using different balanced subsets of \(D_{linchpin\_support}\).<br>
Given a new record \(r_{new} = (a_{new},c_{new})\),  we compose \[r_{i} = a_{new} \oplus c_{i},\  i \in \{top_1, top_2, fusion\}\]<br>
with the format described in section <a href="#input-format">Input Format</a>.<br>
Then, we pass each \(r_i\) through all the \(g\) models of the corresponding care level and display in the UI the mean improvement and decline probability for each \(c_i\).</p>
<h4 id="experiment-files">Experiment files</h4>
<p>Folds where done in our two GPU clusters: panda2 and panda3.<br>
Folds of model \(g\) can be found in panda2 at <em>/workspace/data/ai_core/experiments_results/WAKO_RecordImprovementProbability/v0/*</em><br>
Folds of model \(h\) can be found in panda3 at <em>/workspace/data/ai_core/experiments_results/WAKO_RecordImprovementProbability/v1/*</em></p>
<h4 id="how-to-train-the-models">How to train the models?</h4>
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
