{{ name | escape | underline }}

.. autoclass:: {{ fullname }}
   :show-inheritance:

   {% if '__init__' in methods %}
     {% set caught_result = methods.remove('__init__') %}
   {% endif %}

   {% block attributes_documentation %}
   {% if attributes %}

   {% set collapse_id_suffix = (fullname | replace('.', '-') | replace('_', '-')) %}

   .. raw:: html

      <a class="attr-details-header collapse-header collapsed" data-bs-toggle="collapse" href="#attrDetails-{{ collapse_id_suffix }}" role="button" aria-expanded="false" aria-controls="attrDetails-{{ collapse_id_suffix }}">
         <h2 style="font-size: 24px;">
            <i class="fas fa-angle-down rotate" style="float: right;"></i> Attributes
         </h2>
      </a>
      <div class="collapse show" id="attrDetails-{{ collapse_id_suffix }}">

   {% block attributes_summary %}
   {% if attributes %}

   .. autosummary::
      
   {% for item in attributes %}
      ~{{ name }}.{{ item }}
   {%- endfor %}

   {% endif %}
   {% endblock %}

   {% for item in attributes %}
   .. autoattribute:: {{ item }}
   {%- endfor %}

   .. raw:: html

      </div>

   {% endif %}
   {% endblock %}

   {% block methods_documentation %}
   {% if methods %}

   .. raw:: html

      <a class="meth-details-header collapse-header collapsed" data-bs-toggle="collapse" href="#methDetails-{{ collapse_id_suffix }}" role="button" aria-expanded="false" aria-controls="methDetails-{{ collapse_id_suffix }}">
         <h2 style="font-size: 24px;">
            <i class="fas fa-angle-down rotate" style="float: right;"></i> Methods
         </h2>
      </a>
      <div class="collapse show" id="methDetails-{{ collapse_id_suffix }}">

   {% block methods_summary %}
   {% if methods %}

   .. autosummary::
   {% for item in methods %}
      ~{{ name }}.{{ item }}
   {%- endfor %}

   {% endif %}
   {% endblock %}

   {% for item in methods %}
   .. automethod:: {{ item }}
   {%- endfor %}

   .. raw:: html

      </div>

   {% endif %}
   {% endblock %}

   .. raw:: html

      <script type="text/javascript">
             document.querySelectorAll('.collapse-header').forEach((header) => {
                header.addEventListener('click', () => {
                   const icon = header.querySelector('h2 i');
                   if (icon) icon.classList.toggle('up');
                });
             });
      </script>
