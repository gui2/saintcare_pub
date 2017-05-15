<p><strong>DOCUMENT UNDER EDITION</strong></p>
<h1 id="wako-record-improvement-probability">WAKO Record Improvement Probability</h1>
<p>We have 3 models of Wako,<br>
and 2 models of Saintcare. Each model is split into two.</p>
<h2 id="intro">Intro</h2>
<p>The Wako city dataset is composed of 6497 electronic medical records (EMR). A record is a pair \((a,c)\), where \(a\) is a health assessment (<a href="https://www.evernote.com/shard/s25/nl/2147483647/fd37d473-bea9-41d5-92ec-927ab1f6b6b4/">assessment variables</a>) and \(c\) is a care plan. The assessment is a set of health variables measured by experts and doctors and the care plan is a sequence of health services \(s_{q}\).<br>
A record is also associated to a care level \(\{12,13,21,22,23,24,25\}\) and to an improvement label \(\{improve, maintain, decline\}\).</p>
<h2 id="task">Task</h2>
<p>We aim at learning a model to predict the improvement/decline of a new record. We learn the joint latent representation of care plans and health assessment variables and from this representation we infer the improvement/decline health outcome.</p>
<h2 id="method">Method</h2>
<p>We fit a deep neural network \(\phi: a,c \rightarrow \hat{y}\) with \(\hat{y} \in {1,0}\), care plan \(c\), and assessment \(a\). The model produces a binary classification output, e.i. \(improve = 1\) or \(decline = 0\). We don’t use \(maintain\) records.</p>
<h3 id="wako-assessments-and-careplans">WAKO assessments and careplans</h3>
<p>An assessment contains subparts:<br>
\[a = a_{care~level}^{1} \oplus a_{gender}^{1} \oplus a_{expert}^{79} \oplus a_{doctor}^{94} \oplus a_{diseases}^{596} \oplus c_{services}^{276}\]<br>
where:</p>
<ul>
<li>\(a_{expert}\) contains a vector of 79 health variables provided by experts.</li>
<li>\(a_{doctor}\) contains a vector of 94 health variables measured by an independent medical doctor.</li>
<li>\(a_{diseases}\) contains a binary vector indicating the presence of certain diseases categorized by ICD10 standard. Per record, we can have up to 15 diseases. We have represented the information as a global vector \(a_{diseases}\) of 596 dimensions. Each dimension corresponds to a ICD10 disease of our dataset.  Each variable \(v_d \in a_{diseases}^{1,\dots,596}\) represents the presence or absence of the disease \(d\).</li>
<li>\(c\) contains the care plan services represented as a binary vector  of 276 dimensions, where 276 are all possible services in our dataset. A service is acctivated \(c_{services}^{i}=1\) when appears in the training example.</li>
</ul>
<p><strong>WAKO dataset Stats</strong><br>
Dataset length: 2467 records (after preprocessing)<br>
Assessment dimension: 769<br>
Careplan dimension: 276<br>
Input size: 1045<br>
Vocabulary size: 2064</p>
<p>All values part of the input are categorized following the rules mentioned in section <a href="#preprocessing">Preprocessing</a>.</p>
<h3 id="embedding-layers">Embedding layers</h3>
<p>Our model maps the assesment vector \(a\) and the careplan vector \(c\) into higher dimensional spaces by means of two embedding functions \(l_a\) and \(l_c\). An embedding function  \(l\) maps an input value \(s \in \mathbb{N}\) into a vector through a lookup table operation \(l(s, W) \in \mathbb{R}^k\). The lookup table \(W \in \mathbb{R}^{k}\times |D|\) is learnt, where \(|D|\) is the length of possible v catealues of \(s\) and \(k\) an hyperparameter. For an input vector such as our assessment \(a\), the embedding \(l_a(a,W_a) \in \mathbb{R}^{|a| \times k}\) fits a latent matrix \(W_a \in \mathbb{R}^{k \times |a|}\), and for the careplan vector representation \(c\), \(l_c(c,W_c) \in \mathbb{R}^{k \times |D|}\) we fit a matrix \(W_a \in \mathbb{R}^{k \times |D|}\).</p>
<h3 id="encoder-layers">Encoder layers</h3>
<p>We encode \(l_a\) and \(l_c\) into same dimensional vectors using affine transformations \(f_a: \mathbb{R}^{k \times |a|} \to \mathbb{R}^{100}\), \(f_c: \mathbb{R}^{k \times |c|} \to \mathbb{R}^{100}\). We apply a non linear (RELU) transformation to the output vector \(f_i\).</p>
<h3 id="joint-representation">Joint representation</h3>
<p>Our model guarantees that both assessment and care plan are used for inferring the patient’s health outcome. Otherwise, the model could learn to dismiss the information on the care plan and predicting outcomes using only the assessment information. Such a prediction would be wrong since it would be expressing that the patient can improve with “some” care plan instead of the specific one we are providing as input. The opposite situation is also valid. To generate such a guarantee, we fuse the vector representations corresponding to an assessment \(f_a\) and a care plan \(f_c\) into a feature vector using a bilinear transformation, followed by an affine transform \(\theta\) to obtain the classification output.</p>
<p>\[\phi(a,c) = \theta(f_a ~W ~f_c^T)\]</p>
<p>We optimized the cross entropy loss function \(Y * log(\phi(a,c))\) using stochastic gradient descent.</p>
<h3 id="preprocessing">Preprocessing</h3>
<p>To reduce ambiguities, we categorized all variables and created a dictionary \(D\). Its vocabulary is composed of 2064 words, which correspond to the possible observation values appearing in assessments and care plans. For categorizing the data:</p>
<ul>
<li>We considered empty observations (lack of data) variable dependent. Thus, we assigned a token per variable to represents them.</li>
<li>We considered the cross-modality of the numerical observations –the same observed value have different meaning across variables. To preserve meaning we assign a unique category variable observation.</li>
</ul>
<h2 id="experiments">Experiments</h2>
<p>We aim at predicting the improvement label of a care plan implementation. We set up the task as a binary classification problem where we build a single algorithm to handle all care levels at once. We gathered the records corresponding to \(improve\) and \(decline\) labels.</p>
<p>For easing the understanding of sections below, we describe the experiment making reference to both models structure with the term <em>model</em>.</p>
<h3 id="train--validation">Train / Validation</h3>
<p>Each record \((a,c)\) is associated with an improvement label, where \(a\) is a vector of the assessment variables \(&lt;a_{1},...,a_{769}&gt;\), \(c\) is a set of services.</p>
<p>In our dataset, there are 36 pairs that shares the values of expert variables \(a_{1}\dots a_{79}\). All these records (in total 72) were removed from the set before constructing the dataset.</p>
<p>The difference between \(g\) and \(g'\) relies in the input data. The model \(g\) takes all the variables in an assessment as input. Model \(g'\) takes only the health assessment variables (ie. does not consider the care level and the gender of the record).</p>
<p>Dataset input options:<br>
\(D_{nursing\_care}\) are records within certified with \(\{21,22,23,24,25\}\)<br>
\(D_{linchpin\_support}\) are records certified with \(\{12,13\}\)</p>
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
<h4 id="evaluation">Evaluation</h4>
<p>We measure the Accuracy of our method at classifying the improvement or decline of an (assessment, careplan) input pair. For each experiment we perform cross validation with 10 folds. Each fold was trained during 300 epochs.</p>
<h5 id="data-balancing">Data balancing</h5>
<p>Given that our experiment’s data appears unbalanced (see Table 1), at each fold we reduce the model bias sub-sampling. We <strong>down-sampled</strong> the decline examples to match the number of improve examples, fininshing with a 50/50% split.</p>
<h5 id="trainvalidation-sets">Train/Validation sets</h5>
<p>We construct the validation \((VAL)\) set by extracting 20% of examples from the balanced sets. For validation, the random chance of the model’s Accuracy is 0.5.</p>
<h5 id="results">Results</h5>
<p>In Table 2, we show the number of training/validation examples at each fold and the Mean Max Accuracy the experiments \(D_{nursing\_care}\) and \(D_{linchpin\_support}\).</p>
<table>
<thead>
<tr>
<th>Dataset</th>
<th>Folds</th>
<th>Training examples</th>
<th>Validation examples</th>
<th>\(g\) Mean Max Accuracy (Std)</th>
<th>\(g'\) Mean Max Accuracy (Std)</th>
</tr>
</thead>
<tbody>
<tr>
<td>\(D_{nursing\_care}\)</td>
<td>10</td>
<td>1272</td>
<td>316</td>
<td>0.698 (0.015)</td>
<td>0.762 (0.021)</td>
</tr>
<tr>
<td>\(D_{linchpin\_support}\)</td>
<td>10</td>
<td>154</td>
<td>38</td>
<td>0.621 (0.049)</td>
<td>0.747 (0.049)</td>
</tr>
</tbody>
</table>
<p><strong>Table 2</strong>. Training/validation information and results obtained with models \(g\) and \(g'\). Model \(g\) considers only health variables of the assessment as input, while \(g'\) also considers the care level and gender of the record.</p>
<div>
    <a href="https://plot.ly/~guido.cs.stanford.edu/5560/?share_key=Wp5T1BeyHovugAsUxcr0pd" target="_blank" title="confusion_matrix"><img src="https://plot.ly/~guido.cs.stanford.edu/5560.png?share_key=Wp5T1BeyHovugAsUxcr0pd" alt="confusion_matrix" width="600"></a>
    
</div>
<p><strong>Figure 1</strong>. Confusion matrix of the evaluation of model \(g'\) on a single fold of \(D_{nursiong\_care}\).<br>
Source:<a href="https://plot.ly/~guido.cs.stanford.edu/5560/care-levels-21-22-23-24-25-fold-0/">https://plot.ly/~guido.cs.stanford.edu/5560/care-levels-21-22-23-24-25-fold-0/</a></p>
<div>
    <a href="https://plot.ly/~guido.cs.stanford.edu/5565/?share_key=jlsmCfeGjrJaisxoDbHibW" target="_blank" title="roc"><img src="https://plot.ly/~guido.cs.stanford.edu/5565.png?share_key=jlsmCfeGjrJaisxoDbHibW" alt="roc" width="600"></a>
    
</div>
<p><strong>Figure 2</strong>. ROC curve of model \(g'\) on \(D_{nursiong\_care}\)<br>
Source: <a href="https://plot.ly/~guido.cs.stanford.edu/5565">https://plot.ly/~guido.cs.stanford.edu/5565</a></p>
<div>
    <a href="https://plot.ly/~guido.cs.stanford.edu/5563/?share_key=zuzrTuNZ6rV32i5NsFWAaD" target="_blank" title="confusion_matrix"><img src="https://plot.ly/~guido.cs.stanford.edu/5563.png?share_key=zuzrTuNZ6rV32i5NsFWAaD" alt="confusion_matrix" width="600"></a>
    
</div>
<p><strong>Figure 3</strong>. Confusion matrix of the evaluation of model \(g'\) on a single fold of \(D_{linchpin\_support}\)<br>
Source: <a href="https://plot.ly/~guido.cs.stanford.edu/5563/care-levels-12-13-fold-0/">https://plot.ly/~guido.cs.stanford.edu/5563/care-levels-12-13-fold-0/</a></p>
<div>
    <a href="https://plot.ly/~guido.cs.stanford.edu/5567/?share_key=0o3CYchJKTSuH254Dvc42d" target="_blank" title="roc"><img src="https://plot.ly/~guido.cs.stanford.edu/5567.png?share_key=0o3CYchJKTSuH254Dvc42d" alt="roc" width="600"></a>
    
</div>
<p><strong>Figure 4</strong>. ROC curve of model \(g'\) on \(D_{linchpin\_support}\)<br>
Source: <a href="https://plot.ly/~guido.cs.stanford.edu/5567/improvement-probability-roc-care-levels-12-13/">https://plot.ly/~guido.cs.stanford.edu/5567/improvement-probability-roc-care-levels-12-13/</a></p>
<h4 id="paths-to-the-models-binary-files-in-our-server">Paths to the models’ binary files in our server</h4>
<p>The folds of \(g\) can be found in panda2 at: <br> <em>/workspace/projects/saintcare/src/ai_core/experiments_results/WAKO_RecordImprovementProbability/v0/*</em><br>
The folds of \(g'\) can be found in panda3 at:<br> <em>/workspace/projects/saintcare/src/ai_core/experiments_results/WAKO_RecordImprovementProbability/v5/*</em></p>
<h4 id="production-usage">Production Usage</h4>
<p>Model \(g\) is used in production for estimating the record’s improvement and decline probability of each of the care plans suggested (\(c_{top_1}\), \(c_{top_2}\) and \(c_{fusion}\)) combined with the record’s assessment.<br>
For production, we trained 10 \(g\) models using different balanced subsets of \(D_{nursing\_care}\) and 10 others using different balanced subsets of \(D_{linchpin\_support}\).<br>
Given a new record \(r_{new} = (a_{new},c_{new})\),  we compose \[r_{i} = a_{new} \oplus c_{i},\  i \in \{top_1, top_2, fusion\}\]<br>
with the format described in section <a href="#input-format">Input Format</a>.<br>
Then, we pass each \(r_i\) through all the \(g\) models of the corresponding care level and display in the UI the mean improvement and decline probability for each \(c_i\).</p>
<h3 id="implementation">Implementation</h3>
<p>We use tensorflow with tflearn framework for defining, training and evaluating the model architecture. In our code, the model architecture takes as input the placeholder \(vars\) filled with a matrix of dim <em>d x 1045</em> for model \(g\) (<em>d x 355</em> for model \(h\)) where each row contains the variable values of an assessment \(a\) concatenated with the values of \(c_{services}\).<br>
We defined the target placeholder \(Y_{ph}\) as a matrix of size <em>dx2</em>.</p>
<p>The class <strong>RecordImprovementProbability</strong>, provides methods for training our models. This class is used by the notebook <em>run.ipynb</em>.</p>
<h4 id="how-to-train-your-own-model">How to train your own model?</h4>
<ol>
<li>Open notebook:</li>
<li><code>/workspace/projects/saintcare/src/WAKO\_RecordImprovementProbability/vX/run.ipynb</code></li>
<li>X in [0,1,2,3,4,5]</li>
<li>Configure type of training to do (explained in the notebook).</li>
<li>Save changes on notebook</li>
<li>Go to terminal and run:</li>
</ol>
<pre class=" language-bash"><code class="prism  language-bash"><span class="token function">cd</span> /workspace/projects/saintcare/src/ai_core/WAKO_RecordImprovementProbability/v0
<span class="token comment" spellcheck="true">#generate .py</span>
jupyer nbconvert --to python run.ipynb

<span class="token comment" spellcheck="true">#train model Record Improvement Probability v0</span>
python run.py
</code></pre>
<h4 id="source-code">Source Code</h4>
<p>For model \(g\)<br>
<em>Model estimator</em>: <em>/workspace/projects/saintcare/src/ai_core/WAKO_RecordImprovementProbability/v0/estimator.py</em><br>
<em>Training and evaluation</em>: <em>/workspace/projects/saintcare/src/ai_core/WAKO_RecordImprovementProbability/v0/run.ipynb</em></p>
<p>For model \(g'\):<br>
<em>Model estimator</em>: <em>/workspace/projects/saintcare/src/ai_core/WAKO_RecordImprovementProbability/v5/estimator.py</em><br>
<em>Training and evaluation</em>: <em>/workspace/projects/saintcare/src/ai_core/WAKO_RecordImprovementProbability/v5/run.ipynb</em></p>
<p><strong>Saint Care</strong><br>
Dataset length: 6322 (after removing duplicates)<br>
Assessment size: 200<br>
Careplan size: 828<br>
Input size: 1028<br>
Vocabulary size: 4121</p>
