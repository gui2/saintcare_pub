<h3 id="backbone-architecture">Backbone architecture blablabla</h3>
<p>The blablabla back-end <a href="https://github.com/blusa/ai-production">ai-production</a> is written following a <strong>MVT</strong> programming paradigm in the majority of the code (except few AJAX calls). Since Django allows to create and manage templates itself, we will contemplate *.html files as part of the backend. We will then refer as front-end javascript/jQuery code only.</p>
<p>The source code is based on <a href="http://pinaxproject.com/">pinax boilerplate backbone</a>, so we have certain features being handled by default:</p>
<ul>
<li>Account management turned on</li>
<li>Bootstrap CSS and Javascript pre-linked</li>
<li>Default templates for accounts</li>
</ul>
<h3 id="mvt-design">MVT design</h3>
<p>__For documentation in views, models, forms, etc., run a local instance of the server and access <a href="http://localhost:8000/admin/docs">http://localhost:8000/admin/docs</a> of the server’s branch to develop.</p>
<p>All the functionality not provided by django or pinax boilerplate is implemented under the app <a href="https://github.com/blusa/ai-production/tree/master/production_site/dashboard">dashboard</a>.</p>
<p>Dashboard app follows the structure:</p>
<table>
<thead>
<tr>
<th align="left">Path</th>
<th align="left">Description</th>
</tr>
</thead>
<tbody>
<tr>
<td align="left"><code>forms.py</code></td>
<td align="left">Forms (generally ModelForms) to update models and input data</td>
</tr>
<tr>
<td align="left"><code>urls.py</code></td>
<td align="left">Valid URL’s for dashboard views. (Ajax or not)</td>
</tr>
<tr>
<td align="left"><code>models.py</code></td>
<td align="left">Single, definitive source of information about data</td>
</tr>
<tr>
<td align="left"><code>views.py</code></td>
<td align="left">All of the system views (whether Ajax or not)</td>
</tr>
<tr>
<td align="left"><code>extras.py</code></td>
<td align="left">Mainly code for <strong>AI Plugin API</strong></td>
</tr>
<tr>
<td align="left"><strong><code>templates/</code></strong></td>
<td align="left">This directory and sub-directories contain all <code>*.html</code> files of dashboard</td>
</tr>
<tr>
<td align="left"><strong><code>locale/</code></strong></td>
<td align="left">This directory contains django <code>*.po</code> and <code>*.mo</code> files required for django to translate to <strong>Japanese</strong> (and other languages).</td>
</tr>
<tr>
<td align="left"><strong><code>migrations/</code></strong></td>
<td align="left">Django database migration files</td>
</tr>
</tbody>
</table>
<h4 id="database-models-scheme">Database models scheme</h4>
<p>The system’s database is a relational database based on a PostgreSQL. Tables for both system an original loaded data share the database, but schemes are not consistent between both usages. Original data can be accessed using the <code>Assessment_Plan_Information</code> model.</p>
<p>System models (such as <code>Feedback, AssessmentUIForm</code> models are coded in the same dashboard app).</p>
<p><strong>Insights</strong>:</p>
<p>recommended_careplan</p>
<blockquote>
<p><code>careplan.recommended_careplan</code> exists only to retrieve the assessment corresponding the care plan suggested in retrieval methods</p>
</blockquote>
<p>careplan.category_code</p>
<blockquote>
<p><code>careplan.category_code</code> <strong><em>should not be there</em></strong>. It instead, use function <code>careplan.category_code(self, is_true)</code>.</p>
</blockquote>
<p>assessment.submit_date:</p>
<blockquote>
<p><code>assessment.submit_date</code> in versions \(\leq\) <code>v0.22x</code> represents last edition date: This means, when an assessment is edited, <code>submit_date</code> is updated.</p>
</blockquote>
<p>service_type</p>
<blockquote>
<p><code>service.type</code> is blank when a service is created using the form in the system, but precondition of Excel data is that all services have a type and they are the first two characters of the service code.</p>
</blockquote>
<p>Anyone registered can load a new assessment</p>
<h4 id="ai-integration">AI Integration</h4>
<p>Code for AI integration is in repository <a href="https://github.com/blusa/ai-core">ai-core</a>. API is defined in file <code>extras.py</code> of the <code>dashboard</code> app.</p>
