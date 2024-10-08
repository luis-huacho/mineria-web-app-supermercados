from django.contrib import admin

# Register your models here.
from .models import *

@admin.register(Categoria)
class TaxonomyAdmin(admin.ModelAdmin):
    list_display = ('name', 'market', 'has_children', 'url', 'parent', 'title', 'meta_tag_description')
    search_fields = ('name', 'market', 'has_children', 'url', 'parent', 'title', 'meta_tag_description')
    list_filter = ('name', 'market', 'has_children', 'url', 'parent', 'title', 'meta_tag_description')
    ordering = ('name', 'market', 'has_children', 'url', 'parent', 'title', 'meta_tag_description')