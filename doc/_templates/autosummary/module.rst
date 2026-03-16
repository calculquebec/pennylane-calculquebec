{{ fullname | escape | underline }}

.. automodule:: {{ fullname }}
    :members:
    :undoc-members:
    :inherited-members:
    :show-inheritance:
    :special-members: __init__
    :private-members:

{% if modules %}
Submodules
----------

.. autosummary::
    :toctree:
    :recursive:

{% for item in modules %}
    {{ item }}
{% endfor %}
{% endif %}

{% if classes %}
Classes
-------

.. autosummary::
    :toctree:
    :recursive:

{% for item in classes %}
    {{ item }}
{% endfor %}
{% endif %}

{% if functions %}
Functions
---------

.. autosummary::
    :toctree:
    :recursive:

{% for item in functions %}
    {{ item }}
{% endfor %}
{% endif %}

{% if attributes %}
Module Attributes
-----------------

.. autosummary::
    :toctree:
    :recursive:

{% for item in attributes %}
    {{ item }}
{% endfor %}
{% endif %}
    
